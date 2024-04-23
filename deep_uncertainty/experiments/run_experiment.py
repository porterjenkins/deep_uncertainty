import math
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.loggers import CSVLogger

from deep_uncertainty.enums import DatasetType
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.utils.experiment_utils import fix_random_seed
from deep_uncertainty.utils.experiment_utils import get_chkp_callbacks
from deep_uncertainty.utils.experiment_utils import get_datamodule
from deep_uncertainty.utils.experiment_utils import get_model
from deep_uncertainty.utils.experiment_utils import save_metrics_plots


def main(config: ExperimentConfig):

    fix_random_seed(config.random_seed)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )
    if config.dataset_type == DatasetType.TABULAR:
        datamodule.prepare_data()
        input_dim = datamodule.train_dataloader().dataset.__getitem__(0)[0].size(-1)
    else:
        input_dim = None

    for i in range(config.num_trials):

        model = get_model(config, input_dim)
        chkp_dir = config.chkp_dir / config.experiment_name / f"version_{i}"
        chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)
        logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)

        trainer = L.Trainer(
            accelerator=config.accelerator_type.value,
            min_epochs=config.num_epochs,
            max_epochs=config.num_epochs,
            log_every_n_steps=5,
            check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
            enable_model_summary=True,
            callbacks=chkp_callbacks,
            logger=logger,
        )
        trainer.fit(model=model, datamodule=datamodule)

        metrics = trainer.test(model=model, datamodule=datamodule)[0]
        log_dir = Path(logger.log_dir)
        with open(log_dir / "test_metrics.yaml", "w") as f:
            yaml.dump(metrics, f)
        config.to_yaml(log_dir / "config.yaml")
        save_metrics_plots(log_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ExperimentConfig.from_yaml(args.config))
