from argparse import ArgumentParser
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from matplotlib.pyplot import Axes


ENSEMBLE_HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB Mixture",
        "poisson": "Poisson Mixture",
        "stirn": "Stirn et al. (2023)",
        "seitzer_1.0": r"$\beta_{0.5}$-Gaussian Mixture",
        "immer": "Immer et al. (2024)",
        "seitzer_0.5": r"$\beta_{0.5}$-Gaussian Mixture",
        "beta_ddpn_1.0": r"$\beta_{1.0}$-DDPN Mixture (Ours)",
        "gaussian": "Gaussian Mixture",
        "ddpn": "DDPN Mixture (Ours)",
        "beta_ddpn_0.5": r"$\beta_{0.5}$-DDPN Mixture (Ours)",
    }
)


def produce_figure(save_path: Path | str):
    results_dir = Path("results/reviews/id-ood")
    save_path = Path(save_path)
    palette = sns.color_palette()
    fig, axs = plt.subplots(2, 5, figsize=(10, 4), sharey="all", sharex="all")
    axs: Sequence[Axes]
    for i, (head, name) in enumerate(ENSEMBLE_HEADS_TO_NAMES.items()):
        row_num = i // 5
        col_num = i % 5
        ax: plt.Axes = axs[row_num, col_num]

        log_dir = results_dir / f"{head}_ensemble"
        reg_uncertainties = torch.load(log_dir / "reviews_uncertainties.pt")
        ood_uncertainties = torch.load(log_dir / "bible_uncertainties.pt")
        reg_sum = reg_uncertainties["epistemic"] + reg_uncertainties["aleatoric"]
        ood_sum = ood_uncertainties["epistemic"] + ood_uncertainties["aleatoric"]
        with open(log_dir / "difference_of_means_results.yaml") as f:
            results = yaml.safe_load(f)

        sns.kdeplot(
            reg_sum.detach().cpu().numpy(),
            color=palette[0],
            alpha=0.8,
            label="ID: Amazon Reviews",
            ax=ax,
        )
        sns.kdeplot(
            ood_sum.detach().cpu().numpy(),
            color=palette[1],
            alpha=0.8,
            label="OOD: KJV Bible",
            ax=ax,
        )

        if col_num == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.annotate(f"$\Delta$ = {results['delta']:.3f}", (1, 6.5), fontsize=8)
        ax.annotate(f"$p = {results['p_val']:.3f}$", (1, 5.5), fontsize=8)
        ax.set_title(name, fontsize=9)

    handles, labels = axs.ravel()[-1].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], bbox_to_anchor=(0.9, -0.05), loc="center", fontsize=8)
    fig.text(0.5, -0.01, "Variance", ha="center", fontsize=9)
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
    produce_figure(args.save_path)
