# TODO: Make this match other version.
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson

from deep_uncertainty.evaluation.plotting import plot_posterior_predictive
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.random_variables import DoublePoisson


gt_color = "#e79148"


def produce_figure(
    models: list[DiscreteRegressionNN],
    names: list[str],
    save_path: Path | str,
    dataset_path: Path | str,
):
    """Create a figure showcasing DDPN's ability to fit over/under-dispersed count data.

    Args:
        models (list[DiscreteRegressionNN]): List of models to plot posterior predictive distributions of.
        names (list[str]): List of display names for each respective model in `models`.
        save_path (Path | str): Path to save figure to.
        dataset_path (Path | str): Path with dataset to fit.
    """
    # Change figsize to have a longer height and smaller width
    fig, axs = plt.subplots(1, len(models), figsize=(4.9 * len(models), 2.8), sharey=True)
    axs: Sequence[plt.Axes]
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X = data["X_test"].flatten()
    y = data["y_test"].flatten()

    count = 0

    for model, model_name, ax in zip(models, names, axs):
        if isinstance(model, GaussianNN):
            y_hat = model._predict_impl(torch.tensor(X).unsqueeze(1))
            mu, var = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.flatten().detach().numpy()
            std = var.sqrt().flatten().detach().numpy()
            dist = norm(loc=mu, scale=std)
            nll = np.mean(-np.log(dist.cdf(y + 0.5) - dist.cdf(y - 0.5)))

        elif isinstance(model, DoublePoissonNN):
            y_hat = model._predict_impl(torch.tensor(X).unsqueeze(1))
            mu, phi = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.detach().numpy().flatten()
            phi = phi.detach().numpy().flatten()
            dist = DoublePoisson(mu, phi)
            nll = -np.log(dist.pmf(y)).mean()

        elif isinstance(model, PoissonNN):
            y_hat = model._predict_impl(torch.tensor(X).unsqueeze(1))
            lmbda = y_hat.detach().numpy().flatten()
            dist = poisson(lmbda)
            nll = np.mean(-dist.logpmf(y))

        elif isinstance(model, NegBinomNN):
            y_hat = model._predict_impl(torch.tensor(X).unsqueeze(1))
            mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.flatten().detach().numpy()
            alpha = alpha.flatten().detach().numpy()

            eps = 1e-6
            var = mu + alpha * mu**2
            n = mu**2 / np.maximum(var - mu, eps)
            p = mu / np.maximum(var, eps)
            dist = nbinom(n=n, p=p)
            nll = np.mean(-dist.logpmf(y))

        pred = model._point_prediction(y_hat, training=False).detach().flatten().numpy()
        mae = np.abs(pred - y).mean()
        lower, upper = dist.ppf(0.025), dist.ppf(0.975)

        plot_posterior_predictive(
            X,
            y,
            mu,
            lower=lower,
            upper=upper,
            show=False,
            ax=ax,
            ylims=(0, 45),
            legend=False,
            error_color="gray",
            line_color="#326394",
            line_font=1.5,
            error_alpha=0.6,
            boundary_color="#a9a9a9",
            boundary_width=0.9,
        )

        uncertainty_data = np.load("data/discrete_sine_wave/gt_uncertainty.npz")
        uncertainty_x = uncertainty_data["X"].flatten()
        order = np.argsort(uncertainty_x)
        lb = uncertainty_data["lower"][order]
        ub = uncertainty_data["upper"][order]

        lb_interp = CubicSpline(uncertainty_x[order], lb)
        ub_interp = CubicSpline(uncertainty_x[order], ub)

        foo = np.linspace(uncertainty_x.min(), uncertainty_x.max())
        # here is the color to the dotted line of gt
        ax.plot(foo, lb_interp(foo), "--", c=gt_color)
        ax.plot(foo, ub_interp(foo), "--", c=gt_color)

        ax.set_title(model_name)
        ax.annotate(f"MAE: {mae:.3f}", (0.2, 41))
        ax.annotate(f"NLL: {nll:.3f}", (0.2, 37))

        ax.xaxis.set_major_locator(MultipleLocator(2 * np.pi))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / np.pi:.0f}π")
        )
        ax.set_xlabel(r"$x$", labelpad=-10, fontsize=14, fontname="Times New Roman")
        ax.set_ylabel(None)

        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)

        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

        count += 1

    gt_data = ax.scatter(
        X[0], y[0], facecolors="none", edgecolors="#a9a9a9", alpha=0.4, label="Test data"
    )
    (gt_aleatoric,) = ax.plot(
        foo[0], lb_interp(foo[0]), "--", c=gt_color, label="G.T. Aleatoric", linewidth=1.7
    )
    (learned_mean,) = ax.plot([0], [0], "k", label="Learned Mean")
    learned_aleatoric = ax.fill_between(
        [0], [0], [0], alpha=0.2, color="cornflowerblue", label="Learned Aleatoric"
    )

    fig.text(
        -0.02, 0.5, "y", va="center", rotation="vertical", fontsize=10, fontname="Times New Roman"
    )
    fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles=[gt_data, gt_aleatoric, learned_aleatoric, learned_mean],
        loc="lower center",
        ncol=len(models),
    )

    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/synthetic_demo_iii.pdf"
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    models = [
        GaussianNN.load_from_checkpoint("weights/discrete_sine_wave/gaussian.ckpt"),
        PoissonNN.load_from_checkpoint("weights/discrete_sine_wave/poisson.ckpt"),
        NegBinomNN.load_from_checkpoint("weights/discrete_sine_wave/nbinom.ckpt"),
        DoublePoissonNN.load_from_checkpoint("weights/discrete_sine_wave/ddpn.ckpt"),
    ]
    names = ["Gaussian DNN", "Poisson DNN", "NB DNN", "DDPN (Ours)"]
    produce_figure(models, names, save_path, dataset_path)
