import urllib.request
import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from functools import partial
from typing import Callable, Literal, Optional
from pathlib import Path
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, Imagenette
from data_utils.datasets import shuffled
from data_utils.arrays import take_batches
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


def classification_accuracy(
    net: nn.Module,
    dataloader: DataLoader,
    n_batches: Optional[int] = None,
    device: str = "cpu",
    pred_fn: Optional[Callable] = None,
    predict_runtime: bool = False,
) -> float:
    """Reports the classification accuracy of a network on a given dataset.
    A linear output layer is assumed on the network, with |output_nodes| = |classes|. We calculate
    the softmax and report the top-1 accuracy averaged over the dataloader.
    The Dataloader should return tuples (data, labels) where data is a tensor of shape (batch_size, *input_shape)

    The prediction function should return the probabilitistic output of the network given the input data.

    Evaluation is always done in eval mode, but network is returned to train if it was training before.
    """
    was_training = net.training
    net.train(False)
    try:
        prev_device = next(net.parameters()).device
    except StopIteration:
        print(
            f"Warning: no parameters found to determine previous device with, network will be on device {device} after this operation."
        )
        prev_device = device
    acc = []
    net = net.to(device)
    if n_batches is None:
        n_batches = len(dataloader)
    if predict_runtime:
        now = datetime.now()
    for (x, y), _ in zip(dataloader, range(n_batches)):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            if pred_fn is not None:
                pred = pred_fn(net, x)
            else:
                pred = nn.functional.softmax(net(x), dim=1)

            max_pred = torch.max(pred, dim=1).values
            indices = pred == max_pred.reshape(-1, 1)
            torch.argmax(indices * torch.randn(indices.shape, device=device), dim=1)
            idx = torch.argmax(nn.functional.softmax(pred, dim=1), dim=1)
            acc.extend((idx == y).flatten().cpu())
        if predict_runtime:
            print(
                f"Accuracy will take about {(datetime.now() - now).seconds * n_batches} seconds..."
            )
            predict_runtime = False  # only runs once then
    net = net.to(prev_device)
    net.train(was_training)
    return float(np.mean(acc))


class CvDataset:
    def __init__(self, train_dataset, test_dataset, train_dataloader, test_dataloader):
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def evaluate(
        self,
        model: torch.nn.Module,
        nbatches: int = 1,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
        predict_runtime: bool = False,
    ):
        return classification_accuracy(
            model,
            self.test_dataloader,
            nbatches,
            device=device,
            predict_runtime=predict_runtime,
        )

    def calibration_sample(self, n: int, train: bool = True):
        return take_batches(self.train_dataset, n)[0]


class CIFAR(CvDataset):
    pass


class IMAGENET(CvDataset):
    pass


class COCO(CvDataset):
    pass


class KODAK(CvDataset):
    @property
    def test_dataset(self):
        raise NotImplementedError("Kodak dataset does not have a test dataset.")

    @property
    def test_dataloader(self):
        raise NotImplementedError("Kodak dataset does not have a test dataset.")


class CocoImageFolder(Dataset):
    def __init__(
        self, root: Path, split: Literal["train", "test", "val"], transform=None
    ):
        """
        Custom dataset for loading COCO images without annotations.

        Args:
            root (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.image_paths = []
        for f in (root / f"{split}2017").iterdir():
            if f.suffix in [".png", ".jpg", ".jpeg"]:
                self.image_paths.append(f)
        if len(self.image_paths) == 0:
            raise ValueError("No images found in COCO dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(0)


NormalisationT = tuple[tuple[float, float, float], tuple[float, float, float]]


class Normalisation(Enum):
    CIFAR10_PYTORCH = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    CIFAR10_EDALTOCG = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    CIFAR100_EDALTOCG = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def cifar10(
    shuffle: bool = True,
    normalize: Optional[
        NormalisationT | Normalisation
    ] = Normalisation.CIFAR10_EDALTOCG,
) -> CIFAR:
    return cifar("10", shuffle, normalize=normalize)


def cifar100(
    shuffle: bool = True,
    normalize: Optional[
        NormalisationT | Normalisation
    ] = Normalisation.CIFAR100_EDALTOCG,
) -> CIFAR:
    return cifar("100", shuffle, normalize=normalize)


def cifar(
    kind: Literal["10", "100"] = "10",
    shuffle: bool = True,
    normalize: Optional[NormalisationT | Normalisation] = None,
    batch_size: int = 128,
):
    dataset_class = CIFAR10 if kind == "10" else CIFAR100
    ds_path = os.environ.get("DATASET_PATH", str(Path("~/datasets").expanduser()))

    if normalize is not None:
        if isinstance(normalize, Enum):
            normalize = normalize.value
        assert normalize is not None
        # from https://github.com/kuangliu/pytorch-cifar/issues/19
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize[0], normalize[1]),
            ]
        )
    else:
        transform = transforms.ToTensor()

    cifar_train = dataset_class(
        root=ds_path, train=True, download=True, transform=transform
    )
    cifar_test = dataset_class(
        root=ds_path, train=False, download=True, transform=transform
    )
    if shuffle:
        cifar_train = shuffled(cifar_train)
        cifar_test = shuffled(cifar_test)
    train_dataloader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
    return CIFAR(cifar_train, cifar_test, train_dataloader, test_dataloader)


def imagenet(
    root: str, shuffle: bool = True, transforms=None, batch_size: int = 16
) -> IMAGENET:
    if Path(root).name == "imagenette2":
        Imagenet_class = partial(Imagenette)
    else:
        Imagenet_class = ImageNet

    train_dataset = Imagenet_class(root, split="train", transform=transforms)
    test_dataset = Imagenet_class(root, split="val", transform=transforms)
    if shuffle:
        train_dataset = shuffled(train_dataset)
        test_dataset = shuffled(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return IMAGENET(train_dataset, test_dataset, train_dataloader, test_dataloader)


def coco(
    root: str, shuffle: bool = True, transforms=None, batch_size: int = 16
) -> COCO:
    train_dataset = CocoImageFolder(Path(root), split="train", transform=transforms)
    test_dataset = CocoImageFolder(Path(root), split="val", transform=transforms)
    if shuffle:
        train_dataset = shuffled(train_dataset)
        test_dataset = shuffled(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return COCO(train_dataset, test_dataset, train_dataloader, test_dataloader)


class KodakDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        patch_size: int = 224,
        transform: Optional[Callable] = None,
        download: bool = True,
        force_download: bool = False,
    ):
        download = download or force_download
        if not root_dir.exists() or force_download:
            if not download:
                raise FileNotFoundError(
                    f"Could not find dataset at {root_dir}. "
                    "Set download=True to download the dataset."
                )
            self.download_kodak_dataset(root_dir)
        else:
            print("Found Kodak dataset at ", root_dir)
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.image_paths = [p for p in root_dir.iterdir() if p.suffix == ".png"]
        if not len(self.image_paths) == 24:
            raise ValueError(
                f"Expected 24 images in Kodak dataset, found {len(self.image_paths)}. Dataset may be corrupted, try using force_download=True or remove the dataset at the given path."
            )

    @staticmethod
    def download_kodak_dataset(download_dir: Path = Path("kodak")):
        # from https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite
        print(f"Downloading Kodak dataset to {download_dir}...")
        cnt = 0
        download_dir.mkdir(exist_ok=True, parents=True)
        for im in tqdm(range(1, 25)):
            im_path = download_dir / f"{im:02}.png"
            im_url = f"http://r0k.us/graphics/kodak/kodak/kodim{im:02}.png"
            urllib.request.urlretrieve(im_url, im_path)
            cnt += 1
        print(f"Successfully downloaded {cnt} Kodak Images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Randomly crop a patch of size patch_size x patch_size
        augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        image = augmentation(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(0)


def kodak(
    dataset_dir: Path = Path("~/datasets/kodak"),
    transform: Optional[Callable] = None,
    force_download: bool = False,
    batch_size: int = 8,
):
    train_dataset = KodakDataset(
        root_dir=dataset_dir.expanduser(),
        patch_size=224,
        transform=transform,
        force_download=force_download,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return KODAK(train_dataset, None, train_dataloader, None)
