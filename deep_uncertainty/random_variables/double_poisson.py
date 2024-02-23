import numpy as np
from scipy.stats import loggamma

from deep_uncertainty.random_variables.base import DiscreteRandomVariable


class DoublePoisson(DiscreteRandomVariable):
    def __init__(self, mu: float | int | np.ndarray, phi: float | int | np.ndarray):
        self.mu = np.array(mu)
        self.phi = np.array(phi)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        return np.exp(self.logpmf(x))

    def logpmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the log probability that this random variable takes on the value(s) x.

        Args:
            x (int | float | np.ndarray): The value(s) to compute the log probability of.

        Returns:
            log_probability (float | np.ndarray): The log probability of x.
        """
        x = np.array(x)
        eps = 1e-5
        c = 1 + ((1 - self.phi) / (12 * self.mu * self.phi)) * (1 + (1 / (self.mu * self.phi)))

        return (
            0.5 * np.log(self.phi)
            - self.phi * self.mu
            - np.log(c)
            - x
            + x * np.log(x + eps)
            - loggamma(x + 1)
            + self.phi * x * (1 + np.log(self.mu) - np.log(x + eps))
        )

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
