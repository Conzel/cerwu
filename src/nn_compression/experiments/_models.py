"""
Module containing data models for experiment configuration and results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Compression:
    """Configuration for compression settings."""

    method: str = "optq"
    log10_lm: float = 0
    groupsize: int = -1
    entropy_model: str = "deepcabac"
    nbins: int = 16
    scan_order_major: str = "row"


@dataclass
class Calibration:
    """Configuration for calibration settings."""

    dataset: str = "cifar10"
    nbatches: int = 5000
    batch_size: int = 128


@dataclass
class Evaluation:
    """Configuration for evaluation settings."""

    dataset: str = "cifar10"
    nbatches: int = 5
    batch_size: int = 128
    entropy_model: str = "deepcabac"
    transpose: bool = False


@dataclass
class Network:
    """Network description"""

    name: str = "resnet18"
    train_dataset: str = "cifar10"


@dataclass
class Logger:
    level: str = "info"
    flush: bool = False


@dataclass
class Config:
    """Main configuration for experiments."""

    experiment_name: str = "Experiment"
    experiment_description: str = "No description provided."
    experiment_task: str = "cv"
    device: str = "cpu"
    database_path: str = "./experiments.db"
    compression: Compression = field(default_factory=Compression)
    calibration: Calibration = field(default_factory=Calibration)
    evaluation: Evaluation = field(default_factory=Evaluation)
    network: Network = field(default_factory=Network)
    logger: Logger = field(default_factory=Logger)
    benchmark: bool = True


@dataclass
class OutputResults:
    """Results from running an experiment."""

    performance: float
    bitrate: float

    def __str__(self) -> str:
        return (
            f"Performance: {self.performance:.3f} at {self.bitrate:.3f} bits per weight"
        )


@dataclass
class Experiment:
    """Complete experiment with configuration and results."""

    timestamp: str
    config: Dict[str, Any]
    output_results: OutputResults
