from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
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
    id_color = palette[0]
    ood_color = palette[1]

    versions = range(5)
    domain = np.linspace(0, 3, num=200)

    fig, axs = plt.subplots(
        2,
        len(HEADS_TO_NAMES),
        figsize=(12, 3),
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for col_num, (head, name) in enumerate(HEADS_TO_NAMES.items()):
        deltas = []
        p_vals = []
        reg_quartiles = []
        ood_quartiles = []

        kde_ax: plt.Axes = axs[0, col_num]
        boxplot_ax: plt.Axes = axs[1, col_num]
        kde_ax.tick_params(axis="both", which="both", labelsize=8, grid_color="gray")
        boxplot_ax.tick_params(axis="both", which="both", labelsize=8, grid_color="gray")

        for version in versions:
            log_dir = root_dir / head / f"version_{version}"
            reg_entropies = torch.load(log_dir / "reviews_entropies.pt").detach().cpu().numpy()
            ood_entropies = torch.load(log_dir / "bible_entropies.pt").detach().cpu().numpy()
            reg_quartiles.append(np.quantile(reg_entropies, [0.25, 0.50, 0.75]))
            ood_quartiles.append(np.quantile(ood_entropies, [0.25, 0.50, 0.75]))

            with open(log_dir / "difference_of_means_results.yaml") as f:
                results = yaml.safe_load(f)

            reg_kde = gaussian_kde(reg_entropies, bw_method=0.5)
            ood_kde = gaussian_kde(ood_entropies, bw_method=0.5)
            deltas.append(results["delta"])
            p_vals.append(results["p_val"])
            kde_ax.plot(
                domain,
                reg_kde(domain),
                color=id_color,
                alpha=0.5,
                label="ID: Amazon Reviews",
            )
            kde_ax.plot(
                domain,
                ood_kde(domain),
                color=ood_color,
                alpha=0.5,
                label="OOD: KJV Bible",
            )

        reg_quartiles = np.row_stack(reg_quartiles).mean(axis=0)
        ood_quartiles = np.row_stack(ood_quartiles).mean(axis=0)

        boxplot = boxplot_ax.boxplot(
            x=[
                ood_entropies,
                reg_entropies,
            ],
            vert=False,
            sym="",
            usermedians=[
                ood_quartiles[1],
                reg_quartiles[1],
            ],
            conf_intervals=[
                [ood_quartiles[0], ood_quartiles[2]],
                [reg_quartiles[0], reg_quartiles[2]],
            ],
            patch_artist=True,
        )
        boxplot_ax.set_yticks([])
        boxplot["boxes"][0].set_facecolor(ood_color)
        boxplot["boxes"][1].set_facecolor(id_color)
        for median in boxplot["medians"]:
            median.set_color("black")

        if col_num == 0:
            kde_ax.set_ylabel("Density", fontsize=9)
        kde_ax.set_xlim(-0.1, 2.5)
        kde_ax.annotate(f"$\\bar{{\Delta}}$ = {np.mean(deltas):.3f}", (0, 5.8), fontsize=8)
        kde_ax.annotate(f"$\\bar{{p}} = {np.mean(p_vals):.3f}$", (0, 5.0), fontsize=8)
        kde_ax.set_title(name, fontsize=9)

    for ax in axs.ravel():
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
    handles, labels = kde_ax.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], bbox_to_anchor=(0.9, -0.05), loc="center", fontsize=8)
    fig.text(0.5, -0.04, "Entropy", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root-dir", default="logs/reviews")
    parser.add_argument(
        "--save-path", default="deep_uncertainty/figures/ddpn/artifacts/ood_behavior.pdf"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(args.root_dir, args.save_path)
