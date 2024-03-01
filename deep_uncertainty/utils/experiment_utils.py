from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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
        if config.head_kwargs is not None:
            initializer = partialclass(GaussianNN, **config.head_kwargs)
        else:
            initializer = GaussianNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON:
        if config.head_kwargs is not None:
            initializer = partialclass(DoublePoissonNN, **config.head_kwargs)
        else:
            initializer = DoublePoissonNN

    model = initializer(
        input_dim=1,
        backbone_type=config.backbone_type,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    return model


def get_dataloaders(
    dataset_path: str | Path, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
