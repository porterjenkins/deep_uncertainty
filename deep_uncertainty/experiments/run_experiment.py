from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import (
    CSVLogger,
)  # TODO: This logs locally, but we may want WandB eventually.
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.experiments.enums import HeadType
from deep_uncertainty.experiments.regression_metrics import RegressionMetrics
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import MeanNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN


def get_model(config: ExperimentConfig) -> BaseRegressionNN:
    if config.head_type == HeadType.MEAN:
        initializer = MeanNN
    elif config.head_type == HeadType.GAUSSIAN:
        initializer = GaussianNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON:
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


def get_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = np.load(config.dataset_path)
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
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=9,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=9,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=9,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def main(config: ExperimentConfig):

    # TODO: Implement repeated runs?

    model = get_model(config)
    checkpoint_callback = ModelCheckpoint(
        config.chkp_dir / config.experiment_name, every_n_epochs=100
    )
    logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
    train_loader, val_loader, test_loader = get_dataloaders(config)

    trainer = L.Trainer(
        accelerator=config.accelerator_type.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        log_every_n_steps=25,
        check_val_every_n_epoch=10,
        enable_model_summary=False,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    metrics = RegressionMetrics(**trainer.test(model=model, dataloaders=test_loader)[0])

    print(metrics)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ExperimentConfig.from_yaml(args.config))
