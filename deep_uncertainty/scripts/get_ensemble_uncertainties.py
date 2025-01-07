import os
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from deep_uncertainty.datamodules import BibleDataModule
from deep_uncertainty.datamodules import ReviewsDataModule
from deep_uncertainty.enums import HeadType
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.utils.configs import EnsembleConfig


def get_uncertainties(
    log_dir: Path, config_path: Path, dataset: Literal["amazon-reviews", "bible"]
):
    """Measure the aleatoric/epistemic uncertainty of the provided ensemble on the specified ID / OOD dataset.

    Args:
        log_dir (Path): Directory where results should be logged.
        config_path (Path): Path to ensemble config.
        dataset (str): Either "amazon-reviews" or "bible".
    """
    config = EnsembleConfig.from_yaml(config_path)
    experiment_log_dir = log_dir / config.experiment_name
    if not experiment_log_dir.exists():
        os.makedirs(experiment_log_dir)
    batch_size = config.batch_size
    num_workers = os.cpu_count()
    head_type = config.member_head_type

    if dataset == "amazon-reviews":
        datamodule = ReviewsDataModule(
            root_dir="data/amazon-reviews",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    elif dataset == "bible":
        datamodule = BibleDataModule(
            root_dir="data/bible",
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    if head_type == HeadType.DOUBLE_POISSON:
        model = DoublePoissonMixtureNN.from_config(config)
    elif head_type == HeadType.POISSON:
        model = PoissonMixtureNN.from_config(config)
    elif head_type == HeadType.NEGATIVE_BINOMIAL:
        model = NegBinomMixtureNN.from_config(config)
    else:
        raise ValueError("This experiment is for discrete regression ensembles only.")

    device = model.members[0].device
    aleatoric_uncertainties = []
    epistemic_uncertainties = []
    with torch.inference_mode():
        loop = tqdm(test_loader, desc="Gathering uncertainties...")
        for batch_encoding, _ in loop:
            batch_encoding = batch_encoding.to(device)
            model(batch_encoding)
            uncertainties = model._predict_impl(batch_encoding)[1]
            aleatoric, epistemic = torch.split(uncertainties, [1, 1], dim=1)
            aleatoric_uncertainties.append(aleatoric)
            epistemic_uncertainties.append(epistemic)

        aleatoric_uncertainties = torch.cat(aleatoric_uncertainties).flatten()
        epistemic_uncertainties = torch.cat(epistemic_uncertainties).flatten()
        save_dict = {"aleatoric": aleatoric_uncertainties, "epistemic": epistemic_uncertainties}

        save_path = experiment_log_dir / f"{dataset}_entropies.pt"
        print(f"Saving uncertainties to {save_path}")
        torch.save(save_dict, save_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory to log eval metrics in.",
    )
    parser.add_argument("--config-path", type=Path, help="Path to model config.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon-reviews", "bible"],
        help="The dataset to get entropies from. Either 'amazon-reviews' (in-distribution) or 'bible' (out of distribution).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    get_uncertainties(
        log_dir=args.log_dir,
        config_path=args.config_path,
        dataset=args.dataset,
    )
