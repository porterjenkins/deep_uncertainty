from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import RandomRotation
from torchvision.transforms import ToTensor

from deep_uncertainty.enums import DatasetName
from deep_uncertainty.enums import DatasetType
from deep_uncertainty.enums import HeadType
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import MeanNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.utils.generic_utils import partialclass


def get_model(config: ExperimentConfig) -> BaseRegressionNN:
    if config.head_type == HeadType.MEAN:
        initializer = MeanNN
    elif config.head_type == HeadType.GAUSSIAN:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                GaussianNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = GaussianNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                DoublePoissonNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = DoublePoissonNN

    if config.dataset_type == DatasetType.SCALAR:
        input_dim = 1
        is_scalar = True
    elif config.dataset_type == DatasetType.IMAGE:
        if config.dataset_spec == DatasetName.ROTATED_MNIST:
            input_dim = 1
            is_scalar = False
        else:
            input_dim = 3
            is_scalar = False
    elif config.dataset_type == DatasetType.TABULAR:
        raise NotImplementedError("Tabular data not yet supported.")

    model = initializer(
        input_dim=input_dim,
        is_scalar=is_scalar,
        backbone_type=config.backbone_type,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    return model


def get_dataloaders(
    dataset_type: DatasetType,
    dataset_spec: Path | DatasetName,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if dataset_type == DatasetType.SCALAR:
        data = np.load(dataset_spec)
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

    elif dataset_type == DatasetType.IMAGE:
        if dataset_spec == DatasetName.ROTATED_MNIST:
            transform = Compose([ToTensor(), RandomRotation(45)])
            dataset = MNIST(root="./data/rotated-mnist", download=True, transform=transform)
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                lengths=[0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(1998),  # For reproducibility.
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=9,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=9,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=9,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def save_losses_plot(log_dir: Path):

    metrics = pd.read_csv(log_dir / "metrics.csv")
    train_loss = metrics.iloc[:-1]["train_loss"].dropna()
    val_loss = metrics.iloc[:-1]["val_loss"].dropna()

    fig, ax = plt.subplots(1, 1)
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.legend()
    fig.savefig(log_dir / "losses.png")
    plt.close(fig)
