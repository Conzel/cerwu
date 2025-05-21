"""
Module for calculating and saving layer-wise Hessians for neural networks.
"""

import os
from pathlib import Path

import torch
from nn_compression.networks import LayerWiseHessianTracker
from nn_compression.cv import CvModel
from nn_compression.nlp import LanguageModel
import hydra
from omegaconf import OmegaConf
from nn_compression.experiments import Config, Logger


def calc_hessians(net, dataloader, filepath, nbatches, device="cuda"):
    """
    Calculate Hessians and save them to the given filepath.

    Args:
        net: The neural network model
        dataloader: Dataloader for the calibration dataset
        filepath: Path to save the Hessian file
        nbatches: Number of batches to use
        device: Device to use for computation
    """
    hessian_filepath = Path(filepath)
    batches_so_far = 0
    with LayerWiseHessianTracker(net, save_to=filepath, is_large_net=False):
        while batches_so_far < nbatches:
            for xs, _ in dataloader:
                if batches_so_far >= nbatches:
                    break
                net(xs.to(device))
                batches_so_far += 1
                if batches_so_far % 10 == 0:
                    print(f"{batches_so_far}/{nbatches} batches done", flush=True)

    assert hessian_filepath.exists()
    print("Hessian computed and saved at ", hessian_filepath)
    return hessian_filepath


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: Config) -> None:
    """
    Main function to calculate and save Hessians.

    Args:
        cfg: The configuration loaded by Hydra.
    """
    Logger.init(cfg.logger.level, exist_ok=True)
    logger = Logger.get_instance()

    # Display the configuration
    logger.log("Hydra Configuration:")
    logger.log(OmegaConf.to_yaml(cfg))

    # Determine network type and load
    network_id = cfg.network.name
    if "resnet" in network_id:
        network_id = network_id + "_" + cfg.network.train_dataset

    if cfg.experiment_task == "cv":
        net_enum = CvModel.from_string(network_id)
    elif cfg.experiment_task == "nlp":
        net_enum = LanguageModel.from_string(network_id)
    else:
        raise ValueError(f"Unknown task: {cfg.experiment_task}")

    net = net_enum.load()
    net.to(cfg.device)

    # Create hessians directory if it doesn't exist
    output_dir = Path("hessians")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate hessian filepath with the correct naming convention
    hessian_filepath = (
        output_dir
        / f"{cfg.network.name}_{cfg.network.train_dataset}_{cfg.calibration.nbatches}.pt"
    )

    # Get dataloader for calibration
    ds = net_enum.get_dataset(batch_size=cfg.calibration.batch_size)

    # Calculate Hessians using your existing function
    logger.log(f"Calculating Hessians for {network_id}...")
    calc_hessians(
        net=net,
        dataloader=ds.train_dataloader,
        filepath=hessian_filepath,
        nbatches=cfg.calibration.nbatches,
        device=cfg.device,
    )

    logger.log(f"Hessian calculation completed. Saved to: {hessian_filepath}")

    # Move model back to CPU to free GPU memory
    net.to("cpu")


if __name__ == "__main__":
    main()
