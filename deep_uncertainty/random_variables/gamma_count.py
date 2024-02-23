import numpy as np
from scipy.stats import gamma

from deep_uncertainty.random_variables.base import DiscreteRandomVariable


class GammaCount(DiscreteRandomVariable):
    """A gamma-count random variable (as defined in https://link.springer.com/article/10.1007/s10182-021-00432-6).

    This random variable has support [0, inf) and is parametrized by alpha and beta. Its pmf is given by

    P(X = x) = gamma_cdf(alpha * x, beta) - gamma_cdf(alpha * (x + 1), beta).

    Args:
        alpha (int | float | np.ndarray): Alpha parameter of this gamma-count distribution.
        beta (int | float | np.ndarray): Beta parameter of this gamma-count distribution.
    """

    def __init__(self, alpha: int | float | np.ndarray, beta: int | float | np.ndarray):
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)

        # Pre-compute expected value and variance.
        truncated_support = np.arange(2000)
        self.expected_value = (truncated_support.reshape(-1, 1) * self.pmf(truncated_support)).sum(
            axis=0
        )
        self.variance = (
            truncated_support.reshape(-1, 1) ** 2 * self.pmf(truncated_support)
        ).sum() - self.expected_value**2
        self.standard_deviation = np.sqrt(self.variance)

    def _gamma_cdf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Internal method for computing the gamma cdf as defined by the gamma count distribution.

        Args:
            x (int | np.ndarray): The value(s) to evaluate the gamma cdf at.

        Returns:
            probability (int | np.ndarray): The output of the gamma cdf at x.
        """
        x = np.array(x)
        probability = np.zeros((x.size, self.alpha.size)).squeeze()
        probability[x.squeeze() == 0] = np.ones(self.alpha.size)
        probability[x.squeeze() != 0] = gamma.cdf(
            self.beta, a=(self.alpha * x[x != 0].reshape(-1, 1))
        ).squeeze()
        return probability

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        return self._gamma_cdf(x) - self._gamma_cdf(x + 1)

    def logpmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the log probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the log probability of.

        Returns:
            log_probability (float | np.ndarray): The log probability of x.
        """
        log_probability = np.log(self.pmf(x))
        return log_probability

    def nll(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the negative log likelihood of x for this random variable.

        Args:
            x (int | np.ndarray): The value(s) to compute the negative log likelihood of.

        Returns:
            nll (float | np.ndarray): The negative log likelihood of x.
        """
        return -self.logpmf(x)

    def ppf(self, q: float) -> int | np.ndarray:
        """Return the largest possible value of this random variable at which the probability mass to the left is less than or equal to `q`.

        Args:
            q (float): The desired quantile.

        Returns:
            int | np.ndarray: The largest value at which this distribution has mass <= `q` to the left of it.
        """
        truncated_support = np.arange(2000).reshape(-1, 1)
        mass = self.pmf(truncated_support)
        mass = mass / mass.sum(axis=0)  # Sometimes, the resultant mass isn't entirely normalized.
        mask = np.cumsum(mass, axis=0) <= q
        values = len(mass) - np.argmax(mask[::-1], axis=0) - 1
        values[mask.sum(axis=0) == 0] = 0
        return values.item() if values.size == 1 else values
