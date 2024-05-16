from typing import Callable

import numpy as np
from scipy.stats import entropy
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


def compute_cluster_ece(y: np.ndarray, probs: np.ndarray, cluster_labels: np.ndarray):
    """Estimate the calibration error by clustering in x.

    Given discrete regression targets, probabilistic model output, and labels indicating input cluster membership for each target, compute the calibration error.

    To estimate the calibration error, we cluster in `x` and approximate the empirical distribution within that cluster. We then see how
    close the model's predicted probabilities lie to the empirical distribution within the cluster.

    The `probs` array is assumed to specify the discrete probability that y takes on the values 0 through M (0 << M << infinity).

    Args:
        y (np.ndarray): Discrete regression targets (should be counts). Shape: (n,)
        probs (np.ndarray): For each regression target, the probability that the target takes on the values 0 through M (0 << M << infinity). Shape: (M + 1, n)
        cluster_labels (np.ndarray): Label array indicating which input cluster each regression target is associated with. Shape: (n,)
    """
    eps = 1e-5
    unnormalized_ece = 0
    cluster_sizes = {}
    unique_cluster_indices = np.unique(cluster_labels)

    for cluster_idx in unique_cluster_indices:
        in_cluster = cluster_labels == cluster_idx
        cluster_sizes[cluster_idx] = np.sum(in_cluster)
        y_in_cluster = y[in_cluster]

        cluster_prob_est = np.zeros(len(probs))
        cluster_values, value_counts = np.unique(y_in_cluster, return_counts=True)
        cluster_prob_est[cluster_values] = value_counts / value_counts.sum()

        cluster_prob_est = np.maximum(cluster_prob_est, eps)
        pred_target_probs = np.maximum(probs[:, in_cluster], eps)

        divergences = entropy(pred_target_probs, cluster_prob_est.reshape(-1, 1))
        unnormalized_ece += divergences.sum()

    return unnormalized_ece / len(unique_cluster_indices)


def compute_mcmd(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    x_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lmbda: float = 0.01,
) -> np.ndarray:
    """Given a ground-truth conditional distribution and samples from a model's approximation of that distribution, compute the maximum conditional mean discrepancy (MCMD) along the provided grid.

    Args:
        grid (np.ndarray): Grid of values (assumed to be drawn from X) to compute MCMD across.
        x (np.ndarray): The conditioning values that produced y.
        y (np.ndarray): Ground truth samples from the conditional distribution.
        x_prime (np.ndarray): The conditioning values that produced y_prime.
        y_prime (np.ndarray): Samples from a model's approximation of the ground truth conditional distribution.
        x_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the conditioning variable (x).
        y_kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function to use for the output variable (y).
        lmbda (float, optional): Regularization parameter. Defaults to 0.01.

    Returns:
        np.ndarray: MCMD values along the provided grid.
    """
    n = len(x)
    m = len(x_prime)

    K_X = x_kernel(x.reshape(-1, 1), x.reshape(-1, 1))
    K_X_prime = x_kernel(x_prime.reshape(-1, 1), x_prime.reshape(-1, 1))

    W_X = np.linalg.inv(K_X + n * lmbda * np.eye(n))
    W_X_prime = np.linalg.inv(K_X_prime + m * lmbda * np.eye(m))

    K_Y = y_kernel(y.reshape(-1, 1), y.reshape(-1, 1))
    K_Y_prime = y_kernel(y_prime.reshape(-1, 1), y_prime.reshape(-1, 1))
    K_Y_Y_prime = y_kernel(y.reshape(-1, 1), y_prime.reshape(-1, 1))

    k_X = x_kernel(x.reshape(-1, 1), grid.reshape(-1, 1))
    k_X_prime = x_kernel(x_prime.reshape(-1, 1), grid.reshape(-1, 1))

    first_term = np.diag(k_X.T @ W_X @ K_Y @ W_X.T @ k_X)
    second_term = np.diag(2 * k_X.T @ W_X @ K_Y_Y_prime @ W_X_prime.T @ k_X_prime)
    third_term = np.diag(k_X_prime.T @ W_X_prime @ K_Y_prime @ W_X_prime.T @ k_X_prime)

    return first_term - second_term + third_term


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
