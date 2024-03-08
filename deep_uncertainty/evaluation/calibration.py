import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import rv_continuous


def compute_continuous_ece(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous,
    num_bins: int = 100,
    weights: str = "uniform",
    alpha: float = 1.0,
) -> float:
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), compute the expected calibration error of the model.

    Given a set of probability values {p_1, ..., p_m} spanning [0, 1], and a set of regression targets {y_i | 1 <= i <= n}, the expected calibration error is defined as follows:

        If F_i denotes the posterior predictive cdf for y_i and q_j = |{y_i | F_i(y_i) <= p_j, i = 1, 2, ..., n}| / n, we have

            ECE = sum(w_j * abs(p_j - q_j)^alpha)

        where alpha controls the severity of penalty for a given probability residual.

    Args:
        y_true (np.ndarray): The true values of the regression targets.
        posterior_predictive (RandomVariable): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): The number of bins to use for the ECE. Defaults to 100.
        weights (str, optional): Strategy for choosing the weights in the ECE sum. Must be either "uniform" or "frequency" (terms are weighted by the numerator of q_j). Defaults to "uniform".
        alpha (float, optional): Controls how severely we penalize the model for the distance between p_j and q_j. Defaults to 1 (error term is |p_j - q_j|).

    Returns:
        float: The expected calibration error.
    """
    eps = 1e-5
    p_j = np.linspace(eps, 1 - eps, num=num_bins)
    cdf_less_than_p = posterior_predictive.cdf(y_true) <= p_j.reshape(-1, 1)
    q_j = cdf_less_than_p.mean(axis=1)

    if weights == "uniform":
        w_j = np.ones_like(q_j)
    elif weights == "frequency":
        w_j = cdf_less_than_p.sum(axis=1)
    else:
        raise ValueError(
            f"Weights strategy must be either 'uniform' or 'frequency'. Received {weights}"
        )

    w_j = w_j / w_j.sum()
    ece = np.dot(w_j, np.abs(p_j - q_j) ** alpha)
    return ece


def compute_discrete_ece(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    bin_strategy: str = "adaptive",
    alpha: float = 1.0,
    num_bins: int = 30,
):
    """Given targets and predictions from a discrete probabilistic regression model, compute the expected calibration error.

    Suppose we discretize [0, 1] into a set of m bins and assign each target {y_i | 1 <= i <= n} to a bin based on P(\\hat{y_i}),
    the probability of the model's prediction for that target. Define acc(B) to be the probability, within bin B, that y_i = \\hat{y_i}.
    Define conf(B) to be the average of P(\\hat{y_i}) within bin B. Then we have

        ECE = mean([ |acc(B) - conf(B))|^alpha for B in bins ])

    where alpha controls the severity of the penalty for the magnitude of a given probability residual.

    Bin boundaries can either be selected uniformly across [0, 1] or chosen such that each bin has the same number of targets.

    Args:
        targets (np.ndarray): The regression targets.
        preds (np.ndarray): The model's predictions for the targets (mode of the posterior predictive distribution).
        probs (np.ndarray): The model's probabilities for each of its predictions (probability of the mode of the posterior predictive distribution).
        bin_strategy (str, optional): Strategy for choosing bin boundaries. Must be either "uniform" or "adaptive". Defaults to "adaptive" (same # of targets in each bin).
        alpha (int, optional): Controls how severely we penalize the model for the magnitude of a probability residual. Defaults to 1 (error term is |acc(B) - conf(B)|).
        num_bins (int): The number of bins to use. Defaults to 30.

    Returns:
        float: The expected calibration error.
    """
    if bin_strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, num=num_bins)
        weights = None
    elif bin_strategy == "adaptive":
        bin_boundaries = pd.qcut(probs, num_bins, retbins=True, duplicates="drop")[1]
        n = len(bin_boundaries) - 1
        weights = np.ones(n) / n
    else:
        raise ValueError('Invalid bin strategy specified. Must be "uniform" or "adaptive".')

    mask_matrix = (bin_boundaries[:-1, None] <= probs) & (probs <= bin_boundaries[1:, None])
    bin_counts = mask_matrix.sum(axis=1) + 1e-16
    bin_confidences = np.where(mask_matrix, probs, 0).sum(axis=1) / bin_counts

    if weights is None:
        weights = 1 / bin_counts

    bin_accuracies = np.where(mask_matrix, (preds == targets), 0).sum(axis=1) / bin_counts
    ece = np.dot(weights, np.abs(bin_accuracies - bin_confidences) ** alpha)
    return ece


def compute_young_calibration(y_true: np.ndarray, posterior_predictive: rv_continuous) -> float:
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), compute the Young calibration of the model.

    The Young calibration is defined as 1 minus 4/3 the area between the calibration curve of a perfectly calibrated model (y = x)
    and the given model (the 4/3 multiplier is so that the score lives between 0 and 1)

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive (rv_continuous): Random variable representing the posterior predictive distribution over the targets.
    """
    epsilon = 1e-4
    n = 1000
    p_vals, binwidth = np.linspace(0 + epsilon, 1 - epsilon, num=n, retstep=True)

    area_between_model_and_perfect_calibration_curve = (
        binwidth
        * np.abs(
            p_vals - (posterior_predictive.cdf(y_true) <= p_vals.reshape(-1, 1)).mean(axis=1)
        ).sum()
    )

    calibration_score = 1 - ((4 / 3) * area_between_model_and_perfect_calibration_curve)
    return calibration_score


if __name__ == "__main__":

    # Sanity check.
    x = np.linspace(0, 3, 1000)
    y_true = 3 * x

    # We have nearly perfect calibration here since our model knows its uncertainty is 1.
    y_pred = y_true + np.random.randn(1000)
    sigma_pred = np.ones_like(x)
    posterior = norm(y_pred, sigma_pred)

    # Should be close to 1.
    print(f"Mean Calibration: {compute_young_calibration(y_true, posterior)}")
    print(f"Expected Calibration Error: {compute_continuous_ece(y_true, posterior)}")
