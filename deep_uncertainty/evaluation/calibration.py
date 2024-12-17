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


if __name__ == "__main__":

    # Sanity check.
    x = np.linspace(0, 3, 1000)
    y_true = 3 * x

    # We have nearly perfect calibration here since our model knows its uncertainty is 1.
    y_pred = y_true + np.random.randn(1000)
    sigma_pred = np.ones_like(x)
    posterior = norm(y_pred, sigma_pred)

    print(f"Expected Calibration Error: {compute_continuous_ece(y_true, posterior)}")
