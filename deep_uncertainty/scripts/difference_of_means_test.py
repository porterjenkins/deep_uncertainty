from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import permutation_test
from tqdm import tqdm

from deep_uncertainty.constants import HEADS_TO_NAMES


def difference_of_means(x: np.ndarray, y: np.ndarray, axis: int):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


heads = HEADS_TO_NAMES.keys()
versions = range(5)

for head, version in tqdm(
    product(heads, versions), desc="Running permutation tests...", total=len(heads) * len(versions)
):
    log_dir = Path(f"logs/reviews/{head}/version_{version}")
    reg_entropies = torch.load(log_dir / "reviews_entropies.pt").detach().cpu().numpy()
    ood_entropies = torch.load(log_dir / "bible_entropies.pt").detach().cpu().numpy()

    test_result = permutation_test(
        data=[ood_entropies, reg_entropies],
        statistic=difference_of_means,
        vectorized=True,
        alternative="greater",
        n_resamples=1500,
    )
    results_dict = {"delta": float(test_result.statistic), "p_val": float(test_result.pvalue)}
    with open(log_dir / "difference_of_means_results.yaml", "w") as f:
        yaml.safe_dump(results_dict, f)
