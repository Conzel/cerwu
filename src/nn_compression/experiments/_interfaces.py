import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from enum import Enum
from pathlib import Path
from typing import Literal, Optional
import timm

from nn_compression.cv import CIFAR, IMAGENET, KODAK, COCO
from nn_compression.nlp import Wikitext


class Datasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    @staticmethod
    def _ds_root() -> str:
        return str(Path("~/datasets").expanduser())

    def to_dataset(
        self, split: Literal["train", "test"], dataset_root: Optional[Path] = None
    ) -> Dataset:
        train = split == "train"
        download = True
        transform = torchvision.transforms.ToTensor()
        ds_root = dataset_root or self._ds_root()
        if self == Datasets.CIFAR10:
            return CIFAR10(
                root=str(ds_root),
                train=train,
                download=download,
                transform=transform,
            )
        else:
            return CIFAR100(
                root=str(ds_root),
                train=train,
                download=download,
                transform=transform,
            )

    def to_dataloader(
        self, split: Literal["train", "test"], dataset_root: Optional[Path] = None
    ) -> DataLoader:
        if self == Datasets.CIFAR10 or self == Datasets.CIFAR100:
            batch_size = 128
        else:
            raise ValueError(f"Unknown dataset {self}")
        return DataLoader(
            self.to_dataset(split, dataset_root=dataset_root),
            batch_size=batch_size,
            shuffle=split == "train",
            pin_memory=True,
        )


class Models(Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"

    def to_pretrained_model(
        self,
        dataset: Datasets,
    ) -> nn.Module:
        model = timm.create_model(f"{self.value}_{dataset.value}", pretrained=True)
        for layer in model.children():
            if isinstance(layer, nn.BatchNorm2d):
                layer.track_running_stats = False
        model.train(True)
        return model


DatasetType = Wikitext | CIFAR | IMAGENET | KODAK | COCO

VISION_NETS = [
    "resnet18",
    "resnet50",
    "resnet34",
    "vgg16",
    "vit_b_16",
    "mobilenetv3_large",
    "mobilenetv3_small",
    "efficientnet_b7",
]
NLP_NETS = Literal["gpt2", "gpt2-xl", "pythia-70m", "pythia-1b"]

NETWORK_T = Literal[
    "resnet18",
    "resnet50",
    "resnet34",
    "gpt2",
    "gpt2-xl",
    "vit_b_16",
    "vgg16",
    "gpt2",
    "gpt2-xl",
    "pythia-70m",
    "pythia-1b",
]
DATASET_T = Literal["cifar10", "cifar100", "imagenet", "wikitext2", "kodak", "coco"]
