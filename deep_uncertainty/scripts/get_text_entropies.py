import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import torch
from tqdm import tqdm

from deep_uncertainty.datamodules import BibleDataModule
from deep_uncertainty.datamodules import ReviewsDataModule
from deep_uncertainty.enums import HeadType
from deep_uncertainty.evaluation.utils import calculate_entropy
from deep_uncertainty.experiments.config import EnsembleConfig
from deep_uncertainty.experiments.config import ExperimentConfig
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.experiment_utils import get_model


def main(log_dir: Path, config_path: Path, chkp_path: Path | None, dataset: str):

    is_ensemble = chkp_path is None

    if not log_dir.exists():
        os.makedirs(log_dir)
    config = (
        ExperimentConfig.from_yaml(config_path)
        if not is_ensemble
        else EnsembleConfig.from_yaml(config_path)
    )
    batch_size = config.batch_size
    num_workers = 24
    head_type = config.head_type if not is_ensemble else config.member_head_type

    if dataset == "reviews":
        datamodule = ReviewsDataModule(
            root_dir="./data/reviews",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    elif dataset == "bible":
        datamodule = BibleDataModule(
            root_dir="./data/bible",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
        raise ValueError(f"Expected one of ['reviews', 'bible'] for `dataset` but got {dataset}.")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    if not is_ensemble:
        initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
        model = initializer.load_from_checkpoint(chkp_path)
        device = model.device
    else:
        if head_type == HeadType.GAUSSIAN:
            model = GaussianMixtureNN.from_config(config)
        elif head_type == HeadType.DOUBLE_POISSON:
            model = DoublePoissonMixtureNN.from_config(config)
        elif head_type == HeadType.POISSON:
            model = PoissonMixtureNN.from_config(config)
        elif head_type == HeadType.NEGATIVE_BINOMIAL:
            model = NegBinomMixtureNN.from_config(config)
        device = model.members[0].device

    eps = torch.tensor(1e-6, device=device)

    with torch.inference_mode():
        entropies = []
        max_val = 500
        trunc_support = torch.arange(0, max_val, device=device).reshape(-1, 1)
        loop = tqdm(test_loader, desc="Calculating entropies...")
        for batch_encoding, _ in loop:
            batch_encoding = batch_encoding.to(device)
            y_hat = model._predict_impl(batch_encoding)

            if head_type == HeadType.GAUSSIAN:
                mu, var = torch.split(y_hat, [1, 1], dim=-1)
                mu = mu.flatten()
                var = var.flatten()
                dist = torch.distributions.Normal(mu, var.sqrt())
                probs = dist.cdf(trunc_support + 0.5) - dist.cdf(trunc_support - 0.5)

            if not is_ensemble:

                if head_type == HeadType.DOUBLE_POISSON:
                    mu, phi = torch.split(y_hat, [1, 1], dim=-1)
                    mu = mu.flatten()
                    phi = phi.flatten()
                    dist = DoublePoisson(mu, phi)
                    probs = dist.pmf(trunc_support)
                elif head_type == HeadType.NEGATIVE_BINOMIAL:
                    mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
                    mu = mu.flatten()
                    alpha = alpha.flatten()
                    var = mu + alpha * mu**2
                    p = mu / torch.maximum(var, eps)
                    failure_prob = torch.minimum(1 - p, 1 - eps)
                    n = mu**2 / torch.maximum(var - mu, eps)
                    dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
                    probs = torch.exp(dist.log_prob(trunc_support))
                elif head_type == HeadType.POISSON:
                    lmbda = y_hat
                    dist = torch.distributions.Poisson(rate=lmbda.flatten())
                    probs = torch.exp(dist.log_prob(trunc_support))

            else:

                if head_type in (
                    HeadType.DOUBLE_POISSON,
                    HeadType.POISSON,
                    HeadType.NEGATIVE_BINOMIAL,
                ):
                    probs = y_hat[:, :max_val].permute(1, 0)

            entropies.append(calculate_entropy(probs))

        entropies = torch.cat(entropies)
        torch.save(entropies, log_dir / f"{dataset}_entropies.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to log eval metrics in.",
    )
    parser.add_argument("--config-path", type=str, help="Path to model config.")
    parser.add_argument(
        "--chkp-path",
        type=str,
        default="",
        help="Path to .ckpt where model weights are saved. Do not specify if the model is an ensemble.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to get entropies from. Either 'reviews' (in-distribution) or 'bible' (out of distribution).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=Path(args.log_dir),
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path) if args.chkp_path != "" else None,
        dataset=args.dataset,
    )
