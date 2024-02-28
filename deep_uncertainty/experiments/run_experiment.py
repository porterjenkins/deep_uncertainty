import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.experiments.regression_metrics import RegressionMetrics
from deep_uncertainty.utils.experiment_utils import get_dataloaders
from deep_uncertainty.utils.experiment_utils import get_model
from deep_uncertainty.utils.experiment_utils import save_losses_plot


def main(config: ExperimentConfig):

    train_loader, val_loader, test_loader = get_dataloaders(config.dataset_path, config.batch_size)

    for _ in range(config.num_trials):

        model = get_model(config)
        checkpoint_callback = ModelCheckpoint(
            config.chkp_dir / config.experiment_name, every_n_epochs=100
        )
        logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)

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
        logger.log_metrics(metrics.__dict__)
        log_dir = Path(logger.log_dir)
        config.to_yaml(log_dir / "config.yaml")
        save_losses_plot(log_dir)
        os.remove(log_dir / "hparams.yaml")  # Unused artifact of lightning.


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ExperimentConfig.from_yaml(args.config))
