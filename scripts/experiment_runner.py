"""
Module for running experiments and generating results.
"""

from pathlib import Path
import os

import torch
import tqdm
from nn_compression.coding import nnc_compress
from nn_compression.networks import LayerWiseHessian, LayerWiseHessianTracker
from nn_compression.cv import (
    CvModel,
    unfold_depthwise_convolutions,
    has_depthwise_convolutions,
)
from nn_compression.nlp import LanguageModel
from datetime import datetime
from nn_compression.experiments import (
    Config,
    OutputResults,
    Experiment,
    DatabaseManager,
    Logger,
)
import hydra
from omegaconf import OmegaConf
from nn_compression.quantisation import EntropyQuantisation
from data_utils.experiments import Timer
import torch


def run_experiment(cfg: Config) -> OutputResults:
    """
    Run an experiment with the given configuration.

    Args:
        cfg: The experiment configuration.

    Returns:
        OutputResults: The results of the experiment.
    """
    logger = Logger.get_instance()
    logger.mem_info()
    if cfg.experiment_task == "dummy":
        return dummy_run(cfg)
    network_id = cfg.network.name
    if "resnet" in network_id:
        network_id = network_id + "_" + cfg.network.train_dataset
    if cfg.experiment_task == "cv":
        net_enum = CvModel.from_string(network_id)
    elif cfg.experiment_task == "nlp":
        net_enum = LanguageModel.from_string(network_id)
    else:
        raise ValueError("Unknown task.")
    net = net_enum.load()

    if cfg.benchmark:
        perf = net_enum.get_dataset(batch_size=cfg.evaluation.batch_size).evaluate(net, nbatches=cfg.evaluation.nbatches, device=cfg.device, predict_runtime=True)  # type: ignore
        print(f"BASELINE PERFORMANCE: {perf}")

    if has_depthwise_convolutions(net):
        print(
            "Depthwise Separable Convolutions in the network detected. Unfolding them..."
        )
        net = unfold_depthwise_convolutions(net)
    net.to("cpu")
    logger.mem_info()

    if cfg.compression.method in ["optq", "optq-rd", "cerwu", "rtn"]:
        logger.log("Loading Hessians into model...")
        hessian_path = get_hessian_path(cfg)
        if not hessian_path.exists():
            print("WARNING: NO HESSIANS FOUND, CALCULATING...")
            calculate_hessians(net, net_enum, cfg)
            print("Finished calculating Hessians.")
        else:
            LayerWiseHessian.load_into_model(
                net,
                torch.load(get_hessian_path(cfg), map_location="cpu"),
            )
        logger.mem_info()
        out = gptq_based_quantization(net, net_enum, cfg)
    else:
        assert cfg.compression.method == "nnc"
        out = nnc_compression(net, net_enum, cfg)

    logger.log(out)
    return out


def get_hessian_path(cfg: Config) -> Path:
    base = Path("hessians")
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
    return Path(
        f"hessians/{cfg.network.name}_{cfg.network.train_dataset}_{cfg.calibration.nbatches}.pt"
    ).resolve()


def nnc_compression(
    net: torch.nn.Module, net_enum: CvModel | LanguageModel, cfg: Config
) -> OutputResults:
    with Timer("Quantization"):
        netq, bitrate = nnc_compress(
            net,
            qp=int(cfg.compression.log10_lm),
            only_quantisable=True,
            verification_fn=net_enum.filter_fn(),
            transpose=cfg.compression.scan_order_major == "col",
        )

    with Timer("Evaluation"):
        perf = net_enum.get_dataset(batch_size=cfg.evaluation.batch_size).evaluate(netq, nbatches=cfg.evaluation.nbatches, device=cfg.device, predict_runtime=True)  # type: ignore
        net.to("cpu")

    out = OutputResults(performance=perf, bitrate=bitrate)
    return out


def gptq_based_quantization(
    net, net_enum: CvModel | LanguageModel, cfg: Config
) -> OutputResults:
    q = EntropyQuantisation(cfg.compression.nbins, 10 ** (cfg.compression.log10_lm), cfg.compression.method, cfg.compression.groupsize, cfg.compression.entropy_model, cfg.device, cfg.compression.scan_order_major, net_enum.filter_fn())  # type: ignore
    with Timer("Quantization"):
        netq = q.quantize_network(net)

    with Timer("Evaluation"):
        perf = net_enum.get_dataset(batch_size=cfg.evaluation.batch_size).evaluate(netq, nbatches=cfg.evaluation.nbatches, device=cfg.device, predict_runtime=True)  # type: ignore
        net.to("cpu")
        bitrate = q.estimate_entropy(netq)

    out = OutputResults(performance=perf, bitrate=bitrate)
    return out


def calculate_hessians(
    net: torch.nn.Module, net_enum: CvModel | LanguageModel, cfg: Config
):
    net.to(cfg.device)
    ds = net_enum.get_dataset()

    with Timer("Hessian"):
        with LayerWiseHessianTracker(net, get_hessian_path(cfg), is_large_net=False):
            i = 0
            for x, _ in tqdm.tqdm(ds.train_dataloader, total=cfg.calibration.nbatches):
                net(x.to(cfg.device))
                if i >= cfg.calibration.nbatches:
                    break
                i += 1
    net.to("cpu")


def dummy_run(cfg: Config):
    return OutputResults(cfg.compression.log10_lm + 1, cfg.compression.nbins * 2)


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: Config) -> None:
    """
    Main function to run an experiment and save results.

    Args:
        cfg: The experiment configuration loaded by Hydra.
    """
    Logger.init(cfg.logger.level, exist_ok=True)
    logger = Logger.get_instance()
    # Create database manager
    db_manager = DatabaseManager(cfg.database_path)

    # Display the configuration
    logger.log("Hydra Configuration:")
    logger.log(OmegaConf.to_yaml(cfg))

    # Run experiment
    output_results = run_experiment(cfg)

    # Create Experiment record
    exp = Experiment(
        timestamp=datetime.now().isoformat(),
        # Convert configuration to a dictionary for easy serialization
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        output_results=output_results,
    )

    # Save experiment to the database
    db_manager.save_experiment(exp)
    logger.log("Finished run.")
    if "SLRUM_ARRAY_JOB_ID" in os.environ or "SLURM_JOB_ID" in os.environ:
        if os.environ.get("SLURM_ARRAY_JOB_ID"):
            slurm_job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        else:
            slurm_job_id = os.environ["SLURM_JOB_ID"]

        os.system(f"scancel {slurm_job_id}")


if __name__ == "__main__":
    main()
