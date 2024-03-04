from typing import TypeAlias

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from scipy.stats import rv_discrete

from deep_uncertainty.evaluation.utils import get_bayes_credible_interval
from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


RandomVariable: TypeAlias = rv_discrete | rv_continuous | DiscreteRandomVariable


def plot_posterior_predictive(
    x_test: np.ndarray,
    y_test: np.ndarray,
    preds: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    c: str = "r",
    alpha: float = 0.2,
    show: bool = True,
    title: str = "",
    ax: plt.Axes | None = None,
):
    order = x_test.argsort()

    ax = plt.subplots(1, 1, figsize=(10, 6))[1] if ax is None else ax

    ax.scatter(x_test[order], y_test[order], alpha=0.1, label="Test Data")
    ax.plot(x_test[order], preds[order])
    ax.fill_between(
        x_test[order], lower[order], upper[order], color=c, alpha=alpha, label="95% CI"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(y_test.min() - 5, y_test.max() + 5)
    if show:
        plt.show()


def display_bayes_credible_interval(
    rv: RandomVariable, p: float, ax: plt.Axes | None = None, show=True
) -> tuple[float, float]:
    """Given a univariate random variable and a value `p`, overlay the pmf/pdf with the centered p% Bayes Credible Interval.

    Args:
        rv (RandomVariable): The (univariate) random variable to construct a credible interval for.
        p (float): The probability of the credible interval to construct.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.

    Returns:
        credible_interval_min (float), credible_interval_max (float): The lower and upper bounds of the credible interval.
    """
    credible_interval_min, credible_interval_max = get_bayes_credible_interval(rv, p)
    ax = plt.subplots(1, 1)[1] if ax is None else ax

    if hasattr(rv, "pmf"):
        support = np.arange(rv.ppf(0.001), rv.ppf(0.999) + 1)
        ax.plot(support, rv.pmf(support), color="black", linestyle="-", marker="o")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Support")
        ax.set_ylabel("Probability")
        ax.set_title(f"{p * 100}% credible interval")
        ax.fill_between(
            x=support,
            y1=0,
            y2=rv.pmf(support),
            where=(support >= credible_interval_min) & (support <= credible_interval_max),
            color="green",
            alpha=0.3,
        )

    elif hasattr(rv, "pdf"):
        support = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 300)
        ax.plot(support, rv.pdf(support), color="black")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Support")
        ax.set_ylabel("Density")
        ax.set_title(f"{p * 100}% credible interval")
        ax.fill_between(
            x=support,
            y1=0,
            y2=rv.pdf(support),
            where=(support >= credible_interval_min) & (support <= credible_interval_max),
            color="green",
            alpha=0.3,
        )

    if show:
        plt.show()

    return credible_interval_min, credible_interval_max


def plot_regression_calibration_curve(
    y_true: np.ndarray,
    posterior_predictive: RandomVariable,
    num_bins: int = 9,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """Given targets and a probabilistic regression model (represented as a random variable over the targets), plot a calibration curve.

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive (RandomVariable): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): Specifies how many probability thresholds to use for checking CDF calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    epsilon = 1e-4
    p_vals = np.linspace(0 + epsilon, 1 - epsilon, num=num_bins).reshape(-1, 1)
    expected_pct_where_cdf_less_than_p = p_vals
    actual_pct_where_cdf_less_than_p = (posterior_predictive.cdf(y_true) <= p_vals).mean(axis=1)

    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
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
    )
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
