import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import lightning as L
import yaml

from deep_uncertainty.enums import DatasetType
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.utils.experiment_utils import get_dataloaders
from deep_uncertainty.utils.experiment_utils import get_model


def main(log_dir: Path, config_path: Path, chkp_path: Path):

    if not log_dir.exists():
        os.makedirs(log_dir)
    config = ExperimentConfig.from_yaml(config_path)

    test_loader = get_dataloaders(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )[2]
    if config.dataset_type == DatasetType.TABULAR:
        input_dim = test_loader.dataset.__getitem__(0)[0].size(-1)
    else:
        input_dim = None

    initializer: Type[DiscreteRegressionNN] = get_model(
        config, input_dim, return_initializer=True
    )[1]
    model = initializer.load_from_checkpoint(chkp_path)
    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
    )
    metrics = evaluator.test(model=model, dataloaders=test_loader)[0]
    with open(log_dir / "test_metrics.yaml", "w") as f:
        yaml.dump(metrics, f)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to log eval metrics in.",
    )
    parser.add_argument("--config-path", type=str, help="Path to config.yaml used to train model.")
    parser.add_argument(
        "--chkp-path", type=str, help="Path to .ckpt where model weights are saved."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=Path(args.log_dir),
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path),
    )
