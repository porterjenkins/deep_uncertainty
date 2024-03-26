from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from deep_uncertainty.datasets import CoinCountingDataset
from deep_uncertainty.datasets import ImageDatasetWrapper
from deep_uncertainty.datasets import VEDAIDataset


def get_tabular_npz_train_val_test(
    dataset_path: str | Path,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    train_dataset = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.Tensor(X_val),
        torch.Tensor(y_val).unsqueeze(1),
    )
    test_dataset = TensorDataset(
        torch.Tensor(X_test),
        torch.Tensor(y_test).unsqueeze(1),
    )
    return train_dataset, val_dataset, test_dataset


def get_mnist_train_val_test() -> tuple[Dataset, Subset, Subset]:
    transform = ToTensor()
    train_dataset = MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    val_dataset, test_dataset = random_split(
        MNIST(root="./data/mnist", train=False, download=True, transform=transform),
        lengths=[0.3, 0.7],
    )
    return train_dataset, val_dataset, test_dataset


def get_coin_counting_train_val_test(reduced: bool = True) -> tuple[Subset, Subset, Subset]:
    # Reduced dataset filters all counts higher than 10.
    root_dir = Path(f"../data/coin-counting{'-reduced' if reduced else ''}")
    dataset = CoinCountingDataset(root_dir)

    train_transforms = Compose([Resize((128, 128)), AutoAugment(), ToTensor()])
    inference_transforms = Compose([Resize((128, 128)), ToTensor()])

    train_indices = np.load(root_dir / "train_indices.npy")
    train_dataset = ImageDatasetWrapper(Subset(dataset, train_indices), train_transforms)

    val_indices = np.load(root_dir / "val_indices.npy")
    val_dataset = ImageDatasetWrapper(Subset(dataset, val_indices), inference_transforms)

    test_indices = np.load(root_dir / "test_indices.npy")
    test_dataset = ImageDatasetWrapper(Subset(dataset, test_indices), inference_transforms)

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 9,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def get_vehicles_train_val_test() -> tuple[VEDAIDataset, VEDAIDataset, VEDAIDataset]:
    root_dir = Path("./data/vehicles")

    # TODO: Normalize?
    train_transforms = Compose(
        [Resize((224, 224)), AutoAugment(), ToTensor()]
    )  # May want a larger image size.
    inference_transforms = Compose([Resize((224, 224)), ToTensor()])

    # TODO: Customize which fold?
    train_dataset = VEDAIDataset(root_dir, train=True, fold_num=1, transform=train_transforms)
    test_dataset = VEDAIDataset(root_dir, train=False, fold_num=1, transform=inference_transforms)
    val_dataset = test_dataset

    return train_dataset, val_dataset, test_dataset
