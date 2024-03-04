import numpy as np


class DiscreteRandomVariable:
    """Base class for a discrete random variable."""

    def __init__(self, dimension: int, max_value: int = 2000):
        self.dimension = dimension
        self.max_value = max_value
        self._masses = None
        self._cdf_vals = None

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError("Should be implemented by subclass.")

    def expected_value(self) -> float | np.ndarray:
        raise NotImplementedError("Should be implemented by subclass.")

    def ppf(self, q: float) -> int | float | np.ndarray:
        """Return the smallest possible value of this random variable at which the probability mass to the left is greater than or equal to `q`.

        Args:
            q (float): The desired quantile.

        Returns:
            int | np.ndarray: The smallest value at which this distribution has mass >= `q` to the left of it.
        """
        mask = self.cdf_vals >= min(q, 0.9999)
        lowest_val_with_mass_over_q = np.argmax(mask, axis=0)
        return (
            lowest_val_with_mass_over_q.item()
            if lowest_val_with_mass_over_q.size == 1
            else lowest_val_with_mass_over_q
        )

    def cdf(self, x: int | np.ndarray) -> float | np.ndarray:
        return (
            self.cdf_vals[x].item()
            if self.cdf_vals[x].size == 1
            else self.cdf_vals[x, np.arange(len(x))]
        )

    def rvs(self, size: int | tuple) -> np.ndarray:
        U = np.random.uniform(size=size)
        draws = np.zeros((U.size, self.dimension))
        for i, u in enumerate(U.ravel()):
            draws[i] = self.ppf(u)
        draws = draws.reshape(size, -1)
        return draws

    @property
    def masses(self) -> np.ndarray:
        if self._masses is None:
            truncated_support = np.arange(self.max_value).reshape(-1, 1)
            self._masses = self.pmf(truncated_support)
            self._masses = self._masses / self._masses.sum(axis=0)
        return self._masses

    @property
    def cdf_vals(self):
        if self._cdf_vals is None:
            self._cdf_vals = np.cumsum(self.masses, axis=0)
        return self._cdf_vals
