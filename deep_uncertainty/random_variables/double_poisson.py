import numpy as np
from scipy.special import loggamma
from scipy.special import xlogy

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
        # For numerical stability, we only allow mu to be as small as 1e-6 (and mu/phi to be as small as 1e-4).
        self.mu = np.clip(np.array(mu), a_min=1e-6, a_max=None)
        self.phi = np.array(phi)
        var = np.clip(mu / phi, a_min=1e-4, a_max=None)
        self.phi = self.mu / var

        super().__init__(dimension=self.mu.size, max_value=max_value)

    def expected_value(self) -> float | np.ndarray:
        return self.mu

    def _pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return np.exp(self._logpmf(x))

    def _logpmf(self, x: int | np.ndarray) -> float | np.ndarray:
        x = np.array(x)

        return (
            0.5 * np.log(self.phi)
            - self.phi * self.mu
            - x
            + xlogy(x, x)
            - loggamma(x + 1)
            + self.phi * (x + xlogy(x, self.mu) - xlogy(x, x))
        )
