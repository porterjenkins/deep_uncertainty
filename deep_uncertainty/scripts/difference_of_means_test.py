from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from scipy.stats import permutation_test


def difference_of_means(x: np.ndarray, y: np.ndarray, axis: int):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def main(head: Literal["ddpn", "beta_ddpn_0.5", "beta_ddpn_1.0", "poisson", "nbinom"]):
    log_dir = Path(f"logs/ood/{head}_ensemble")
    reg_uncertainties = torch.load(log_dir / "reviews_uncertainties.pt")
    ood_uncertainties = torch.load(log_dir / "bible_uncertainties.pt")

    reg_aleatoric = reg_uncertainties["aleatoric"].detach().cpu().numpy()
    reg_epistemic = reg_uncertainties["epistemic"].detach().cpu().numpy()
    ood_aleatoric = ood_uncertainties["aleatoric"].detach().cpu().numpy()
    ood_epistemic = ood_uncertainties["epistemic"].detach().cpu().numpy()

    reg_sum = reg_aleatoric + reg_epistemic
    ood_sum = ood_aleatoric + ood_epistemic

    test_result = permutation_test(
        data=[ood_sum, reg_sum],
        statistic=difference_of_means,
        vectorized=True,
        alternative="greater",
        n_resamples=1500,
    )
    results_dict = {"delta": float(test_result.statistic), "p_val": float(test_result.pvalue)}
    with open(log_dir / "difference_of_means_results.yaml", "w") as f:
        yaml.safe_dump(results_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--head", choices=["ddpn", "beta_ddpn_0.5", "beta_ddpn_1.0", "poisson", "nbinom"])
    args = parser.parse_args()
    main(args.head)
