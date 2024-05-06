from typing import TypeAlias

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
