from typing import TypeAlias

import torch
from scipy.stats import rv_continuous
from scipy.stats import rv_discrete

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


RandomVariable: TypeAlias = rv_discrete | rv_continuous | DiscreteRandomVariable


def get_bayes_credible_interval(rv: RandomVariable, p: float) -> tuple[float, float]:
    """Given a random variable `rv` and a value `p`, return the (centered) Bayes Credible Interval in which the random variable falls with probability `p`.

    Args:
        rv (Random Variable): The random variable to construct a credible interval for.
        p (float): The probability of the credible interval to construct.

    Returns:
        credible_interval_min (float), credible_interval_max (float): The lower and upper bounds of the credible interval.
    """
    p_low = (1 - p) / 2
    p_high = p_low + p
    credible_interval_min, credible_interval_max = rv.ppf(p_low), rv.ppf(p_high)

    return credible_interval_min, credible_interval_max


def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the given discrete distribution(s) in native Pytorch.

    If `probs` do not sum to 1 along dimension 0, they will first be normalized.

    Args:
        probs (torch.Tensor): Probabilities that define the discrete distribution(s). Shape: (num_probabilities, num_distributions).

    Returns:
        torch.Tensor: The entropy of the given distribution(s). Shape: (num_distributions,).
    """
    normalized_probs = probs / probs.sum(dim=0)
    return -torch.xlogy(normalized_probs, normalized_probs).sum(dim=0)
