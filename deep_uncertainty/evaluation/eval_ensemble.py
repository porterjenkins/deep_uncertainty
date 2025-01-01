import os
from argparse import ArgumentParser
from argparse import Namespace

import lightning as L
import yaml

from deep_uncertainty.enums import HeadType
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import FaithfulGaussianMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import HomoscedasticGaussianMixtureNN
from deep_uncertainty.models.ensembles import LogFaithfulGaussianMixtureNN
from deep_uncertainty.models.ensembles import LogGaussianMixtureNN
from deep_uncertainty.models.ensembles import MultiClassNNEnsemble
from deep_uncertainty.models.ensembles import NaturalGaussianMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.utils.configs import EnsembleConfig
from deep_uncertainty.utils.experiment_utils import get_datamodule


def main(config_path: str):

    config = EnsembleConfig.from_yaml(config_path)
    log_dir = config.log_dir / config.experiment_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_spec,
        config.batch_size,
    )

    if config.member_head_type == HeadType.GAUSSIAN:
        ensemble = GaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.LOG_GAUSSIAN:
        ensemble = LogGaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.HOMOSCEDASTIC_GAUSSIAN:
        ensemble = HomoscedasticGaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.FAITHFUL_GAUSSIAN:
        ensemble = FaithfulGaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.LOG_FAITHFUL_GAUSSIAN:
        ensemble = LogFaithfulGaussianMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.NATURAL_GAUSSIAN:
        ensemble = NaturalGaussianMixtureNN.from_config(config)
    elif config.member_head_type in (HeadType.DOUBLE_POISSON, HeadType.DOUBLE_POISSON_GLM):
        ensemble = DoublePoissonMixtureNN.from_config(config)
    elif config.member_head_type in (HeadType.POISSON, HeadType.POISSON_GLM):
        ensemble = PoissonMixtureNN.from_config(config)
    elif config.member_head_type in (HeadType.NEGATIVE_BINOMIAL, HeadType.NEGATIVE_BINOMIAL_GLM):
        ensemble = NegBinomMixtureNN.from_config(config)
    elif config.member_head_type == HeadType.MULTI_CLASS:
        ensemble = MultiClassNNEnsemble.from_config(config)
    else:
        raise NotImplementedError(f"Haven't implemented ensemble for {config.member_head_type}.")

    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
        devices=1,
        num_nodes=1,
    )
    metrics = evaluator.test(model=ensemble, datamodule=datamodule)[0]
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
