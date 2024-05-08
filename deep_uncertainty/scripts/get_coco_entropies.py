import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import torch
from tqdm import tqdm

from deep_uncertainty.datamodules import COCOCowsDataModule
from deep_uncertainty.datamodules import COCOPeopleDataModule
from deep_uncertainty.enums import HeadType
from deep_uncertainty.evaluation.utils import calculate_entropy
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.experiment_utils import get_model


def main(log_dir: Path, config_path: Path, chkp_path: Path, split: str):

    if not log_dir.exists():
        os.makedirs(log_dir)
    config = TrainingConfig.from_yaml(config_path)
    batch_size = config.batch_size
    num_workers = 16

    if split == "people":
        datamodule = COCOPeopleDataModule(
            root_dir="./data/coco_people",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    elif split == "cows":
        datamodule = COCOCowsDataModule(
            root_dir="./data/coco_cows",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
        raise ValueError(f"Expected one of ['people', 'cows'] for `split` but got {split}.")
    datamodule.prepare_data()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
    model = initializer.load_from_checkpoint(chkp_path)
    eps = torch.tensor(1e-6, device=model.device)

    with torch.inference_mode():
        entropies = []
        trunc_support = torch.arange(0, 2000, device=model.device).reshape(-1, 1)
        loop = tqdm(test_loader, desc="Calculating entropies...")
        for images, _ in loop:
            y_hat = model._predict_impl(images)
            if config.head_type == HeadType.GAUSSIAN:
                mu, var = torch.split(y_hat, [1, 1], dim=-1)
                mu = mu.flatten()
                var = var.flatten()
                dist = torch.distributions.Normal(mu, var.sqrt())
                probs = dist.cdf(trunc_support + 0.5) - dist.cdf(trunc_support - 0.5)
            elif config.head_type == HeadType.DOUBLE_POISSON:
                mu, phi = torch.split(y_hat, [1, 1], dim=-1)
                mu = mu.flatten()
                phi = phi.flatten()
                dist = DoublePoisson(mu, phi)
                probs = dist.pmf(trunc_support)
            elif config.head_type == HeadType.NEGATIVE_BINOMIAL:
                mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
                mu = mu.flatten()
                alpha = alpha.flatten()
                var = mu + alpha * mu**2
                p = mu / torch.maximum(var, eps)
                failure_prob = torch.minimum(1 - p, 1 - eps)
                n = mu**2 / torch.maximum(var - mu, eps)
                dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
                probs = torch.exp(dist.log_prob(trunc_support))
            elif config.head_type == HeadType.POISSON:
                lmbda = y_hat
                dist = torch.distributions.Poisson(rate=lmbda.flatten())
                probs = torch.exp(dist.log_prob(trunc_support))
            entropies.append(calculate_entropy(probs))

        entropies = torch.cat(entropies)
        torch.save(entropies, log_dir / f"coco_{split}_entropies.pt")


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
    parser.add_argument(
        "--split",
        type=str,
        help="The COCO split to get entropies from. Either 'people' or 'cows'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=Path(args.log_dir),
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path),
        split=args.split,
    )
