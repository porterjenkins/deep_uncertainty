from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import rv_continuous

from deep_uncertainty.evaluation.calibration import compute_continuous_ece
from deep_uncertainty.random_variables import DoublePoisson


def plot_posterior_predictive(
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    error_color: str = "r",
    error_alpha: float = 0.2,
    show: bool = True,
    title: str = "",
    ax: plt.Axes | None = None,
):
    """Visualize a model's posterior predictive distribution over a 1d dataset (`x`, `y` both scalars) by showing the expected value and error bounds across the regression targets.

    Args:
        x (np.ndarray): The x values (inputs).
        y (np.ndarray): The ground truth y values (outputs).
        mu (np.ndarray): The expected values of the model's posterior predictive distribution over `y`.
        upper (np.ndarray): Upper bounds for the model's posterior predictive distribution over `y`.
        lower (np.ndarray): Lower bounds for the model's posterior predictive distribution over `y`.
        error_color (str, optional): Color with which to fill the model's error bounds. Defaults to "r".
        alpha (float, optional): Transparency value for the model's error bounds. Defaults to 0.2.
        show (bool, optional): Whether/not to show the resultant plot. Defaults to True.
        title (str, optional): If specified, a title for the resultant plot. Defaults to "".
        ax (plt.Axes | None, optional): If given, the axis on which to plot the posterior predictive distribution. Defaults to None (axis is created).
    """
    order = x.argsort()

    ax = plt.subplots(1, 1, figsize=(10, 6))[1] if ax is None else ax

    ax.scatter(x[order], y[order], alpha=0.1, label="Test Data", s=3)
    ax.plot(x[order], mu[order])
    ax.fill_between(
        x[order], lower[order], upper[order], color=error_color, alpha=error_alpha, label="95% CI"
    )
    ax.set_title(title)
    ax.set_ylim(y.min() - 5, y.max() + 5)
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
        plt.show()


def plot_regression_calibration_curve_cdf(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous,
    num_bins: int = 9,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), plot a calibration curve.

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive (rv_continuous): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): Specifies how many probability thresholds to use for checking CDF calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    epsilon = 1e-4
    p_vals = np.linspace(0 + epsilon, 1 - epsilon, num=num_bins).reshape(-1, 1)
    expected_pct_where_cdf_less_than_p = p_vals
    actual_pct_where_cdf_less_than_p = (posterior_predictive.cdf(y_true) <= p_vals).mean(axis=1)

    ece = compute_continuous_ece(
        y_true, posterior_predictive, num_bins=50, weights="frequency", alpha=2
    )

    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        expected_pct_where_cdf_less_than_p,
        linestyle="--",
        color="red",
        label="Perfectly calibrated",
    )
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        actual_pct_where_cdf_less_than_p,
        marker="o",
        linestyle="-",
        color="black",
        label="Model",
        markersize=3,
    )
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.annotate(f"ECE: {ece:2f}", xy=(-0.05, 0.9), fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if show:
        plt.show()


def produce_figure(save_path: str | Path):
    """Produce a figure showing four perfectly calibrated predictive distributions with differing ECE.

    Args:
        save_path (str | Path): The path to save the figure at.
    """
    num_samples = 2500
    cont_x = np.random.uniform(1, 10, size=num_samples)
    mean = cont_x
    variance = cont_x
    gaussian_y = norm.rvs(loc=mean, scale=np.sqrt(variance))
    poisson_y = poisson.rvs(mu=mean)
    double_poisson_y = DoublePoisson(mean, 1).rvs((1, num_samples)).flatten()
    eps = 1e-1
    nbinom_y = nbinom.rvs(
        n=(mean**2 / (variance + eps - mean)).round(), p=(mean / (variance + eps))
    )

    mu_hat = mean
    gaussian_post_pred = norm(loc=mean, scale=np.sqrt(variance))

    lambda_hat = mean
    poisson_post_pred = poisson(mu=mean)

    dpo_mu_hat = mean
    dpo_post_pred = DoublePoisson(mean, mean / variance)

    nbinom_mu_hat = mean
    nbinom_post_pred = nbinom(
        n=(mean**2 / (variance + eps - mean)).round(), p=(mean / (variance + eps))
    )

    fig, axs = plt.subplots(3, 4, figsize=(7, 4))
    hist_alpha = 0.6
    hist_rwidth = 0.9

    plot_posterior_predictive(
        cont_x,
        gaussian_y,
        mu_hat,
        *gaussian_post_pred.ppf([[0.025], [0.975]]),
        ax=axs[0, 0],
        show=False,
    )
    axs[1, 0].hist(
        gaussian_post_pred.cdf(gaussian_y), density=True, alpha=hist_alpha, rwidth=hist_rwidth
    )
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    plot_regression_calibration_curve_cdf(gaussian_y, gaussian_post_pred, ax=axs[2, 0], show=False)

    plot_posterior_predictive(
        cont_x,
        poisson_y,
        lambda_hat,
        *poisson_post_pred.ppf([[0.025], [0.975]]),
        ax=axs[0, 1],
        show=False,
    )
    axs[1, 1].hist(
        poisson_post_pred.cdf(poisson_y), density=True, alpha=hist_alpha, rwidth=hist_rwidth
    )
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    plot_regression_calibration_curve_cdf(poisson_y, poisson_post_pred, ax=axs[2, 1], show=False)

    plot_posterior_predictive(
        cont_x,
        double_poisson_y,
        dpo_mu_hat,
        dpo_post_pred.ppf(0.025),
        dpo_post_pred.ppf(0.975),
        ax=axs[0, 2],
        show=False,
    )
    axs[1, 2].hist(
        dpo_post_pred.cdf(double_poisson_y.reshape(-1, 1).flatten()),
        density=True,
        alpha=hist_alpha,
        rwidth=hist_rwidth,
    )
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    plot_regression_calibration_curve_cdf(
        double_poisson_y.flatten(), dpo_post_pred, ax=axs[2, 2], show=False
    )

    plot_posterior_predictive(
        cont_x,
        nbinom_y,
        nbinom_mu_hat,
        nbinom_post_pred.ppf(0.025),
        nbinom_post_pred.ppf(0.975),
        ax=axs[0, 3],
        show=False,
    )
    axs[1, 3].hist(
        nbinom_post_pred.cdf(nbinom_y), density=True, alpha=hist_alpha, rwidth=hist_rwidth
    )
    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])
    plot_regression_calibration_curve_cdf(nbinom_y, nbinom_post_pred, ax=axs[2, 3], show=False)

    row_labels = ["Posterior Predictive", "PIT", "Reliability Diagram"]
    for ax, row in zip(axs[:, 0], row_labels):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            fontsize=6,
            ha="right",
            va="center",
            rotation=90,
        )

    col_labels = ["Gaussian", "Poisson", "Double Poisson", "Negative Binomial"]
    for ax, col in zip(axs[0, :], col_labels):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 15),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=8,
            ha="center",
            va="baseline",
        )

    for ax in axs.ravel():
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    fig.tight_layout()
    fig.savefig(fname=save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/failure_of_pit.pdf"
    produce_figure(save_path)
