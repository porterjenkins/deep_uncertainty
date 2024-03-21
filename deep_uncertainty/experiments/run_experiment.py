import math
import random
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from deep_uncertainty.enums import DatasetType
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.utils.experiment_utils import get_dataloaders
from deep_uncertainty.utils.experiment_utils import get_model
from deep_uncertainty.utils.experiment_utils import save_losses_plot


def main(config: ExperimentConfig):

    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )
    if config.dataset_type == DatasetType.TABULAR:
        input_dim = train_loader.dataset.__getitem__(0)[0].size(-1)
    else:
        input_dim = None

    for i in range(config.num_trials):

        model = get_model(config, input_dim)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.chkp_dir / config.experiment_name / f"version_{i}",
            every_n_epochs=config.chkp_freq,
            filename="{epoch}",
            save_top_k=-1,
        )
        logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)

        trainer = L.Trainer(
            accelerator=config.accelerator_type.value,
            min_epochs=config.num_epochs,
            max_epochs=config.num_epochs,
            log_every_n_steps=math.ceil(len(train_loader) / 2),
            check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
            enable_model_summary=False,
            callbacks=[checkpoint_callback],
            logger=logger,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        metrics = trainer.test(model=model, dataloaders=test_loader)[0]
        logger.log_metrics(metrics)
        log_dir = Path(logger.log_dir)
        config.to_yaml(log_dir / "config.yaml")
        save_losses_plot(log_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ExperimentConfig.from_yaml(args.config))
