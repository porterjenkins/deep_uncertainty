from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from deep_uncertainty.enums import DatasetType
from deep_uncertainty.enums import HeadType
from deep_uncertainty.enums import ImageDatasetName
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import MeanNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.backbones import CNN
from deep_uncertainty.models.backbones import MNISTCNN
from deep_uncertainty.models.backbones import ScalarMLP
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.utils.data_utils import get_coin_counting_train_val_test
from deep_uncertainty.utils.data_utils import get_rotated_mnist_train_val_test
from deep_uncertainty.utils.data_utils import get_scalar_npz_train_val_test
from deep_uncertainty.utils.data_utils import get_train_val_test_loaders
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
        backbone = ScalarMLP()
    elif config.dataset_type == DatasetType.IMAGE:
        if config.dataset_spec == ImageDatasetName.ROTATED_MNIST:
            backbone = MNISTCNN()
        else:
            backbone = CNN()
    elif config.dataset_type == DatasetType.TABULAR:
        raise NotImplementedError("Tabular data not yet supported.")

    model = initializer(
        backbone=backbone,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    return model


def get_dataloaders(
    dataset_type: DatasetType,
    dataset_spec: Path | ImageDatasetName,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    if dataset_type == DatasetType.SCALAR:
        train_dataset, val_dataset, test_dataset = get_scalar_npz_train_val_test(dataset_spec)

    elif dataset_type == DatasetType.IMAGE:
        if dataset_spec == ImageDatasetName.ROTATED_MNIST:
            train_dataset, val_dataset, test_dataset = get_rotated_mnist_train_val_test()
        elif dataset_spec == ImageDatasetName.COIN_COUNTING:
            train_dataset, val_dataset, test_dataset = get_coin_counting_train_val_test()

    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
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
