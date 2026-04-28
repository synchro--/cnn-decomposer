"""Small torchvision dataset helpers for TensorPress examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split


@dataclass(frozen=True, slots=True)
class VisionDatasetSpec:
    """Metadata for a supported small computer vision dataset.

    Parameters
    ----------
    name : str
        CLI-friendly dataset identifier.
    display_name : str
        Human-readable dataset name.
    num_classes : int
        Number of target classes.
    channels : int
        Native image channel count.
    image_size : int
        Native square image size.
    description : str
        Short description for ``--list-datasets`` output.
    """

    name: str
    display_name: str
    num_classes: int
    channels: int
    image_size: int
    description: str


SUPPORTED_DATASETS: dict[str, VisionDatasetSpec] = {
    "cifar10": VisionDatasetSpec(
        "cifar10",
        "CIFAR-10",
        10,
        3,
        32,
        "60k 32x32 color images across 10 object classes.",
    ),
    "cifar100": VisionDatasetSpec(
        "cifar100",
        "CIFAR-100",
        100,
        3,
        32,
        "60k 32x32 color images across 100 fine-grained classes.",
    ),
    "mnist": VisionDatasetSpec(
        "mnist",
        "MNIST",
        10,
        1,
        28,
        "70k 28x28 grayscale handwritten digit images.",
    ),
    "fashion-mnist": VisionDatasetSpec(
        "fashion-mnist",
        "Fashion-MNIST",
        10,
        1,
        28,
        "70k 28x28 grayscale clothing item images.",
    ),
    "kmnist": VisionDatasetSpec(
        "kmnist",
        "KMNIST",
        10,
        1,
        28,
        "70k 28x28 grayscale Kuzushiji character images.",
    ),
    "svhn": VisionDatasetSpec(
        "svhn",
        "SVHN",
        10,
        3,
        32,
        "32x32 real-world cropped house-number images.",
    ),
    "stl10": VisionDatasetSpec(
        "stl10",
        "STL-10",
        10,
        3,
        96,
        "96x96 color image recognition dataset; examples use labeled splits only.",
    ),
}


def supported_dataset_names() -> list[str]:
    """Return supported dataset names in stable CLI order."""
    return list(SUPPORTED_DATASETS)


def format_supported_datasets() -> str:
    """Return a human-readable table of supported datasets."""
    rows = ["Supported datasets:"]
    for spec in SUPPORTED_DATASETS.values():
        rows.append(
            f"  {spec.name:<13} {spec.display_name:<14} "
            f"{spec.channels}x{spec.image_size}x{spec.image_size:<3} "
            f"{spec.num_classes:>3} classes  {spec.description}"
        )
    return "\n".join(rows)


def get_dataset_spec(name: str) -> VisionDatasetSpec:
    """Return metadata for a supported dataset.

    Parameters
    ----------
    name : str
        Dataset identifier.

    Returns
    -------
    VisionDatasetSpec
        Dataset metadata.

    Raises
    ------
    ValueError
        If ``name`` is not supported.
    """
    normalized = name.lower()
    try:
        return SUPPORTED_DATASETS[normalized]
    except KeyError as exc:
        options = ", ".join(supported_dataset_names())
        raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {options}") from exc


def build_transforms(
    spec: VisionDatasetSpec,
    *,
    image_size: int | None = None,
    train: bool = False,
    output_channels: int | None = None,
) -> T.Compose:
    """Build simple transforms for classification examples."""
    size = image_size or spec.image_size
    transforms: list[Any] = []
    if size != spec.image_size:
        transforms.append(T.Resize(size))
    if output_channels == 3 and spec.channels == 1:
        transforms.append(T.Grayscale(num_output_channels=3))
    if train and size >= 32:
        transforms.extend([T.RandomCrop(size, padding=4), T.RandomHorizontalFlip()])
    transforms.extend([T.ToTensor(), T.Normalize((0.5,) * (output_channels or spec.channels), (0.5,) * (output_channels or spec.channels))])
    return T.Compose(transforms)


def _make_dataset(
    name: str,
    *,
    data_dir: str,
    train: bool,
    transform: Any,
    download: bool,
) -> torch.utils.data.Dataset:
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(data_dir, train=train, download=download, transform=transform)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(data_dir, train=train, download=download, transform=transform)
    if name == "mnist":
        return torchvision.datasets.MNIST(data_dir, train=train, download=download, transform=transform)
    if name == "fashion-mnist":
        return torchvision.datasets.FashionMNIST(
            data_dir, train=train, download=download, transform=transform
        )
    if name == "kmnist":
        return torchvision.datasets.KMNIST(data_dir, train=train, download=download, transform=transform)
    if name == "svhn":
        return torchvision.datasets.SVHN(
            data_dir, split="train" if train else "test", download=download, transform=transform
        )
    if name == "stl10":
        return torchvision.datasets.STL10(
            data_dir, split="train" if train else "test", download=download, transform=transform
        )
    raise ValueError(f"Unsupported dataset: {name}")


def _subset(dataset: torch.utils.data.Dataset, size: int | None) -> torch.utils.data.Dataset:
    if size is None or size >= len(dataset):
        return dataset
    return Subset(dataset, range(size))


def get_vision_dataloaders(
    dataset: str,
    *,
    data_dir: str = "./data",
    batch_size: int = 128,
    image_size: int | None = None,
    output_channels: int | None = None,
    train_size: int | None = 2048,
    val_size: int = 512,
    test_size: int | None = 1024,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[dict[str, DataLoader], VisionDatasetSpec]:
    """Create train/validation/test dataloaders for a supported vision dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier, for example ``"cifar10"`` or ``"fashion-mnist"``.
    data_dir : str, default="./data"
        Directory where torchvision stores downloaded datasets.
    batch_size : int, default=128
        Dataloader batch size.
    image_size : int, optional
        Resize images to this square size.
    output_channels : int, optional
        Convert grayscale datasets to this channel count when needed.
    train_size : int, optional
        Maximum number of training samples to use.
    val_size : int, default=512
        Validation split size.
    test_size : int, optional
        Maximum number of test samples to use.
    num_workers : int, default=0
        Dataloader worker count.
    download : bool, default=True
        If True, let torchvision download missing data.

    Returns
    -------
    tuple[dict[str, DataLoader], VisionDatasetSpec]
        Dataloaders and dataset metadata.
    """
    spec = get_dataset_spec(dataset)
    channels = output_channels or spec.channels
    train_transform = build_transforms(
        spec, image_size=image_size, train=True, output_channels=channels
    )
    eval_transform = build_transforms(
        spec, image_size=image_size, train=False, output_channels=channels
    )

    train_dataset = _make_dataset(
        spec.name, data_dir=data_dir, train=True, transform=train_transform, download=download
    )
    val_dataset = _make_dataset(
        spec.name, data_dir=data_dir, train=True, transform=eval_transform, download=download
    )
    test_dataset = _make_dataset(
        spec.name, data_dir=data_dir, train=False, transform=eval_transform, download=download
    )

    n_val = min(val_size, max(len(train_dataset) - 1, 1))
    train_split, _ = random_split(
        train_dataset,
        [len(train_dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    _, val_split = random_split(
        val_dataset,
        [len(val_dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            _subset(train_split, train_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            _subset(val_split, val_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            _subset(test_dataset, test_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders, spec
