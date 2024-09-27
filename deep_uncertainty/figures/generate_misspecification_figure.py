from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from scipy.stats import nbinom

from deep_uncertainty.evaluation.plotting import plot_posterior_predictive
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.figure_utils import multiple_formatter


def produce_figure(
    models: list[DiscreteRegressionNN],
    names: list[str],
    save_path: Path | str,
    dataset_path: Path | str,
):
    """Create a figure showcasing DDPN's ability to fit count data, even when the underlying distribution is not Double Poisson.

    Args:
        models (list[DiscreteRegressionNN]): List of models to plot posterior predictive distributions of.
        names (list[str]): List of display names for each respective model in `models`.
        save_path (Path | str): Path to save figure to.
        dataset_path (Path | str): Path with dataset to fit.
    """
    fig, axs = plt.subplots(1, len(models), figsize=(2.5 * len(models), 3), sharey=True)
    axs: Sequence[plt.Axes]
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X = data["X_test"].flatten()
    y = data["y_test"].flatten()

    for model, model_name, ax in zip(models, names, axs):

        if isinstance(model, DoublePoissonNN):
            y_hat = model._predict_impl(torch.tensor(X).float().unsqueeze(1))
            mu, phi = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.detach().numpy().flatten()
            phi = phi.detach().numpy().flatten()
            dist = DoublePoisson(mu, phi)
            nll = -np.log(dist.pmf(y)).mean()

        elif isinstance(model, NegBinomNN):
            y_hat = model._predict_impl(torch.tensor(X).float().unsqueeze(1))
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
            error_color="cornflowerblue",
        )
        ax.set_title(model_name)
        ax.annotate(f"MAE: {mae:.3f}", (0.2, 41))
        ax.annotate(f"NLL: {nll:.3f}", (0.2, 37))
        ax.xaxis.set_major_locator(MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(FuncFormatter(multiple_formatter()))
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/misspecification_recovery.pdf"
    dataset_path = "data/nbinom_bowtie/nbinom_bowtie.npz"
    weights_dir = Path("weights/nbinom_bowtie")
    models = [
        NegBinomNN.load_from_checkpoint(weights_dir / "nbinom.ckpt"),
        DoublePoissonNN.load_from_checkpoint(weights_dir / "ddpn.ckpt"),
        DoublePoissonNN.load_from_checkpoint(weights_dir / "beta_ddpn_0.5.ckpt"),
        DoublePoissonNN.load_from_checkpoint(weights_dir / "beta_ddpn_1.0.ckpt"),
    ]
    names = ["NB DNN", "DDPN (Ours)", r"$\beta_{0.5}$-DDPN (Ours)", r"$\beta_{1.0}$-DDPN (Ours)"]
    produce_figure(models, names, save_path, dataset_path)
