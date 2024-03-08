import numpy as np
from scipy.special import loggamma

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DoublePoisson(DiscreteRandomVariable):
    """A Double Poisson random variable.

    Attributes:
        mu (float | int | np.ndarray): The mu parameter of the distribution (can be vectorized).
        phi (float | int | np.ndarray): The phi parameter of the distribution (can be vectorized).
        max_value (int, optional): The highest value to assume support for (since the DPO support is infinite). For numerical purposes. Defaults to 2000.
        dimension (int): The dimension of this random variable (matches the dimension of mu and phi).
    """

    def __init__(
        self, mu: float | int | np.ndarray, phi: float | int | np.ndarray, max_value: int = 2000
    ):
        self.mu = np.array(mu)
        self.phi = np.array(phi)
        super().__init__(dimension=self.mu.size, max_value=max_value)

    def expected_value(self) -> float | np.ndarray:
        return self.mu

    def _pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return np.exp(self._logpmf(x))

    def _logpmf(self, x: int | np.ndarray) -> float | np.ndarray:
        x = np.array(x)
        eps = 1e-6

        return (
            0.5 * np.log(self.phi)
            - self.phi * self.mu
            - x
            + x * np.log(np.maximum(x, eps))
            - loggamma(x + 1)
            + self.phi * x * (1 + np.log(self.mu) - np.log(np.maximum(x, eps)))
        )
