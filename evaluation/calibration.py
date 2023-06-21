import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from scipy.stats import norm, truncnorm, binom
from scipy.integrate import quad
import warnings


def get_bayes_credible_interval(rv, p: float) -> Tuple[float, float]:
    """Given a random variable `rv` and a value `p`, return the (centered) Bayes Credible Interval in which the random variable falls with probability `p`.
    
    Args:
        rv: The random variable to construct a credible interval for.
        p (float): The probability of the credible interval to construct.

    Returns:
        confidence_interval_min (float), confidence_interval_max (float): The lower and upper bounds of the credible interval
    """
    p_low = (1 - p) / 2
    p_high = p_low + p
    credible_interval_min, credible_interval_max = rv.ppf(p_low), rv.ppf(p_high)

    return credible_interval_min, credible_interval_max

def display_bayes_credible_interval(rv, p: float):
    interval_min, interval_max = get_bayes_credible_interval(rv, p)
    support = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
    plt.plot(support, rv.pdf(support), color='black')
    plt.ylim(bottom=0)
    plt.vlines(x=[interval_min, interval_max], ymin=[0, 0], ymax=[rv.pdf(interval_min), rv.pdf(interval_max)], colors=['red', 'red'])
    plt.show()
    

def get_pct_of_targets_in_pred_credible_interval(y_true: np.ndarray, posterior_predictive_distribution, p: float = 0.95) -> float:
    """Return the percentage of ground truth targets that are within the p% credible interval of a probabilistic regression model.

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive_distribution: Random variable representing the posterior predictive distribution over the targets.

    Returns:
        pct_of_targets_in_pred_credible_interval (float): The percentage of ground truth targets that are within the specified confidence interval of the model.
    """
    credible_interval_min, credible_interval_max = get_bayes_credible_interval(posterior_predictive_distribution, p)
    target_is_in_credible_interval = (y_true >= credible_interval_min) & (y_true <= credible_interval_max)
    pct_of_targets_in_pred_credible_interval = target_is_in_credible_interval.mean()

    return pct_of_targets_in_pred_credible_interval

def plot_regression_calibration_curve(y_true: np.ndarray, posterior_predictive_distribution, num_bins: int = 9,
                                      ax: Optional[plt.Axes] = None, show: bool = True) -> None:
    """Given targets and a probabilistic regression model (represented as a random variable over the targets), plot a calibration curve.
    
    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive_distribution: Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): Specifies how many probability thresholds to use for checking credible interval calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    epsilon = 1e-4
    credible_interval_probabilities = np.linspace(0 + epsilon, 1 - epsilon, num=num_bins)
    expected_pct_of_targets_in_pred_credible_intervals = credible_interval_probabilities
    actual_pct_of_targets_in_pred_credible_intervals = [get_pct_of_targets_in_pred_credible_interval(y_true, posterior_predictive_distribution, p) for p in credible_interval_probabilities]
         
    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Expected percentage of targets in p% confidence interval")
    ax.set_ylabel("Percentage of targets in predicted p% confidence interval")
    ax.plot(expected_pct_of_targets_in_pred_credible_intervals, expected_pct_of_targets_in_pred_credible_intervals, linestyle='--', color='red', label='Perfectly calibrated')
    ax.plot(expected_pct_of_targets_in_pred_credible_intervals, actual_pct_of_targets_in_pred_credible_intervals, marker='o', linestyle='-', color='black', label='Model')
    ax.legend()
    plt.tight_layout()
    
    if show:
        plt.show()

def compute_average_calibration_score(y_true: np.ndarray, posterior_predictive_distribution) -> float:
    """Given targets and a probabilistic regression model (represented as a random variable over the targets), compute the average calibration score of the model.

    The average calibration score is defined as 1 minus twice the area between the calibration curve of a perfectly calibrated model (y = x)
    and the given model (the 2x multiplier is so that the score lives between 0 and 1)
    
    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive_distribution: Random variable representing the posterior predictive distribution over the targets.
    """
    with warnings.catch_warnings():     # Scipy passes back an annoying integration warning that doesn't affect the output.
        warnings.simplefilter("ignore")
        abs_distance_between_model_and_perfect_calibration = lambda p: abs(get_pct_of_targets_in_pred_credible_interval(y_true, posterior_predictive_distribution, p) - p)
        area_between_model_and_perfect_calibration_curve = quad(abs_distance_between_model_and_perfect_calibration, 0, 1)[0]

    calibration_score = 1 - (2 * area_between_model_and_perfect_calibration_curve)
    
    return calibration_score


if __name__ == "__main__":

    # Sanity check.
    x = np.linspace(0, 3, 1000)
    y_true = 3 * x

    y_pred = y_true + np.random.randn(1000)
    sigma_pred = 3 * np.ones_like(x)    # We have nearly perfect calibration here since our model knows its uncertainty is 1.
    posterior_predictive_distribution = norm(y_pred, sigma_pred)

    plot_regression_calibration_curve(y_true, posterior_predictive_distribution, num_bins=9)
    print(compute_average_calibration_score(y_true, posterior_predictive_distribution))    # Should be close to 1.

    display_bayes_credible_interval(binom(3, 0.5), p=0.9)