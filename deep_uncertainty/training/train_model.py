import math
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.experiment_utils import fix_random_seed
from deep_uncertainty.utils.experiment_utils import get_chkp_callbacks
from deep_uncertainty.utils.experiment_utils import get_datamodule
from deep_uncertainty.utils.experiment_utils import get_model


def main(config: TrainingConfig):

    fix_random_seed(config.random_seed)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )

    for i in range(config.num_trials):

        model = get_model(config)
        chkp_dir = config.chkp_dir / config.experiment_name / f"version_{i}"
        chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)
        logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)

        trainer = L.Trainer(
            accelerator=config.accelerator_type.value,
            min_epochs=config.num_epochs,
            max_epochs=config.num_epochs,
            log_every_n_steps=5,
            check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
            enable_model_summary=False,
            callbacks=chkp_callbacks,
            logger=logger,
            precision=config.precision,
        )
        trainer.fit(model=model, datamodule=datamodule)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(TrainingConfig.from_yaml(args.config))
