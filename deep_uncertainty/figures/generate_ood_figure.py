from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.pyplot import Axes
from scipy.stats import gaussian_kde
from seaborn import color_palette

from deep_uncertainty.constants import HEADS_TO_NAMES


def produce_figure(root_dir: Path | str, save_path: Path | str):
    """Create and save a figure showing the entropy distributions of each Amazon Reviews model, both in and out of distribution.

    Args:
        root_dir (Path | str): Directory where entropy logs are stored.
        save_path (Path | str): The path to save the figure to.
    """
    root_dir = Path(root_dir)
    save_path = Path(save_path)
    palette = color_palette()

    versions = range(5)
    domain = np.linspace(0, 3, num=200)

    fig, axs = plt.subplots(1, len(HEADS_TO_NAMES), figsize=(12, 2), sharey="row", sharex="row")
    axs: Sequence[Axes]
    for col_num, (head, name) in enumerate(HEADS_TO_NAMES.items()):
        deltas = []
        p_vals = []
        for version in versions:
            log_dir = root_dir / head / f"version_{version}"
            reg_entropies = torch.load(log_dir / "reviews_entropies.pt").detach().cpu().numpy()
            ood_entropies = torch.load(log_dir / "bible_entropies.pt").detach().cpu().numpy()
            with open(log_dir / "difference_of_means_results.yaml") as f:
                results = yaml.safe_load(f)

            reg_kde = gaussian_kde(reg_entropies, bw_method=0.5)
            ood_kde = gaussian_kde(ood_entropies, bw_method=0.5)
            deltas.append(results["delta"])
            p_vals.append(results["p_val"])
            axs[col_num].plot(
                domain,
                reg_kde(domain),
                color=palette[0],
                alpha=0.5,
                label="ID: Amazon Reviews",
            )
            axs[col_num].plot(
                domain,
                ood_kde(domain),
                color=palette[1],
                alpha=0.5,
                label="OOD: KJV Bible",
            )

        if col_num == 0:
            axs[col_num].set_ylabel("Density", fontsize=9)
        axs[col_num].annotate(f"$\\bar{{\Delta}}$ = {np.mean(deltas):.3f}", (0, 5.8), fontsize=8)
        axs[col_num].annotate(f"$\\bar{{p}} = {np.mean(p_vals):.3f}$", (0, 5.0), fontsize=8)
        axs[col_num].set_title(name, fontsize=9)
        axs[col_num].set_xticklabels(axs[col_num].get_xticklabels(), fontsize=8)
        axs[col_num].set_yticklabels(axs[col_num].get_yticklabels(), fontsize=8)

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], bbox_to_anchor=(0.9, -0.05), loc="center", fontsize=8)
    fig.text(0.5, -0.04, "Entropy", ha="center", fontsize=9)
    fig.tight_layout()

    fig.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root-dir", default="logs/reviews")
    parser.add_argument(
        "--save-path", default="deep_uncertainty/figures/artifacts/ood_behavior.pdf"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(args.root_dir, args.save_path)
