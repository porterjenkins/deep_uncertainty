from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import RandomRotation
from torchvision.transforms import ToTensor


def get_scalar_npz_train_val_test(
    dataset_path: str | Path,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    train_dataset = TensorDataset(
        torch.Tensor(X_train.reshape(-1, 1)),
        torch.Tensor(y_train.reshape(-1, 1)),
    )
    val_dataset = TensorDataset(
        torch.Tensor(X_val.reshape(-1, 1)),
        torch.Tensor(y_val.reshape(-1, 1)),
    )
    test_dataset = TensorDataset(
        torch.Tensor(X_test.reshape(-1, 1)),
        torch.Tensor(y_test.reshape(-1, 1)),
    )
    return train_dataset, val_dataset, test_dataset


def get_rotated_mnist_train_val_test() -> tuple[Subset, Subset, Subset]:
    transform = Compose([ToTensor(), RandomRotation(45)])
    dataset = MNIST(root="./data/rotated-mnist", download=True, transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        lengths=[0.8, 0.1, 0.1],
    )
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
