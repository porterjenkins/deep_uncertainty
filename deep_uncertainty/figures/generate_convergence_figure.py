from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from deep_uncertainty.models import DoublePoissonNN


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def main(save_path: str | Path):

    data_dir = Path("data/isolated")
    weights_dir = Path("weights/isolated_count_data")

    data = np.load(data_dir / "isolated_count_data.npz")
    X = data["X_train"]
    y = data["y_train"]

    order = np.argsort(X)
    grid = np.linspace(X.min(), X.max(), num=200)
    grid_tensor = torch.tensor(grid).float().unsqueeze(1)
    f = CubicSpline(X[order], X[order] * np.sin(X[order]) + 15)
    epochs = (199, 399, 599, 799, 999)
    colors = sns.dark_palette("#69d", reverse=True, n_colors=len(epochs))
    alpha_vals = np.linspace(0.3, 1.0, num=len(epochs))
    model_heads = "ddpn", "beta_ddpn_0.5", "beta_ddpn_1.0"
    titles = "$\\beta = 0$", "$\\beta = 0.5$", "$\\beta = 1.0$"

    fig, axs = plt.subplots(2, len(model_heads), figsize=(12, 6), sharey="row", sharex="col")
    for j, (model_head, title) in enumerate(zip(model_heads, titles)):
        axs[0, j].set_title(title)
        if j == 0:
            axs[0, j].set_ylabel(r"$\mu$")
            axs[1, j].set_ylabel(r"$\gamma$")

        axs[1, j].set_xlabel("$x$")
        axs[0, j].scatter(X[:-2], y[:-2], color="cornflowerblue", alpha=0.4, s=4)
        axs[0, j].scatter(X[-2:], y[-2:], color="cornflowerblue", alpha=0.5, zorder=10)
        axs[0, j].plot(grid, f(grid), linestyle="dotted", color="black", linewidth=3, zorder=9)
        axs[1, j].plot(
            grid, (6 - 0.03 * grid**2), linestyle="dotted", color="black", linewidth=3, zorder=9
        )

        for epoch, color, alpha in zip(epochs, colors, alpha_vals):
            model = DoublePoissonNN.load_from_checkpoint(
                weights_dir / model_head / f"epoch={epoch}.ckpt"
            )
            mu_hat, phi_hat = torch.split(model.predict(grid_tensor), [1, 1], dim=-1)
            axs[0, j].plot(grid, mu_hat.flatten().detach(), alpha=alpha, color=color, label=epoch)
            axs[1, j].plot(grid, phi_hat.flatten().detach(), alpha=alpha, color=color, label=epoch)

        axs[0, j].set_ylim(0, 50)
        axs[1, j].set_ylim(0.01, 125)
        axs[1, j].set_yscale("log")
        for i in range(2):
            max_y = axs[i, 0].get_ylim()[1]
            axs[i, j].fill_between(
                [X.min() - 0.1, X.min() + 0.1], 0, max_y, color="lightgray", alpha=0.5
            )
            axs[i, j].fill_between(
                [X.max() - 0.1, X.max() + 0.1], 0, max_y, color="lightgray", alpha=0.5
            )
            axs[i, j].fill_between(
                [X[order[1]] - 0.1, X[order[-2]] + 0.1], 0, max_y, color="lightgray", alpha=0.5
            )

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.legend(custom_lines, [x + 1 for x in epochs], title="Epoch", loc="center right")
    fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    main(save_path=Path("deep_uncertainty/figures/artifacts/convergence_diff_beta.pdf"))
