from argparse import ArgumentParser
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import yaml
import seaborn as sns
from matplotlib.pyplot import Axes


ENSEMBLE_HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB Mixture",
        "poisson": "Poisson Mixture",
        "seitzer_0.5": r"$\beta_{0.5}$-Gaussian Mixture",
        "ddpn": "DDPN Mixture (Ours)",
        "beta_ddpn_0.5": r"$\beta_{0.5}$-DDPN Mixture (Ours)",
        "beta_ddpn_1.0": r"$\beta_{1.0}$-DDPN Mixture (Ours)",
    }
)


def produce_figure(save_path: Path | str):
    results_dir = Path("results/reviews/id-ood")
    save_path = Path(save_path)
    palette = sns.color_palette()
    fig, axs = plt.subplots(
        1, 6, figsize=(12, 2), sharey="row", sharex="row"
    )
    axs: Sequence[Axes]
    for col_num, (head, name) in enumerate(ENSEMBLE_HEADS_TO_NAMES.items()):
        log_dir = results_dir / f"{head}_ensemble"
        reg_uncertainties = torch.load(log_dir / "reviews_uncertainties.pt")
        ood_uncertainties = torch.load(log_dir / "bible_uncertainties.pt")
        reg_sum = reg_uncertainties["epistemic"] + reg_uncertainties["aleatoric"]
        ood_sum = ood_uncertainties["epistemic"] + ood_uncertainties["aleatoric"]
        with open(log_dir / "difference_of_means_results.yaml") as f:
            results = yaml.safe_load(f)

        sns.kdeplot(reg_sum.detach().cpu().numpy(), color=palette[0], alpha=0.8, label="ID: Amazon Reviews", ax=axs[col_num])
        sns.kdeplot(ood_sum.detach().cpu().numpy(), color=palette[1], alpha=0.8, label="OOD: KJV Bible", ax=axs[col_num])

        if col_num == 0:
            axs[col_num].set_ylabel("Density", fontsize=9)
        axs[col_num].annotate(f"$\Delta$ = {results['delta']:.3f}", (1, 2.1), fontsize=8)
        axs[col_num].annotate(f"$p = {results['p_val']:.3f}$", (1, 1.8), fontsize=8)
        axs[col_num].set_title(name, fontsize=9)

    handles, labels = axs[-1].get_legend_handles_labels()
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
