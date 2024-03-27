import os
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
import yaml

from deep_uncertainty.enums import HeadType
from deep_uncertainty.experiments.config import EnsembleConfig
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.utils.experiment_utils import get_dataloaders


def main(config_path: str):

    config = EnsembleConfig.from_yaml(config_path)
    log_dir = config.log_dir / config.experiment_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    test_loader = get_dataloaders(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )[2]

    if config.member_head_type == HeadType.GAUSSIAN:
        ensemble = GaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.DOUBLE_POISSON:
        ensemble = DoublePoissonMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.POISSON:
        ensemble = PoissonMixtureNN.from_config(config)
    else:
        raise NotImplementedError(f"Haven't implemented ensemble for {config.member_head_type}.")

    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
    )
    metrics = evaluator.test(model=ensemble, dataloaders=test_loader)[0]
    with open(log_dir / "test_metrics.yaml", "w") as f:
        yaml.dump(metrics, f)
    config.to_yaml(log_dir / "ensemble_config.yaml")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to ensemble config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
