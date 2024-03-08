import numpy as np


class DiscreteRandomVariable:
    """Base class for a discrete random variable, assumed to have support [0, infinity] (truncated at `max_value`)."""

    def __init__(self, dimension: int, max_value: int = 2000):
        self.dimension = dimension
        self.max_value = max_value
        self._pmf_vals = None
        self._cdf_vals = None

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        d = self.pmf_vals.shape[1]
        result: np.ndarray = self.pmf_vals[x, np.arange(d)]
        if result.size == 1:
            return result.item()
        return result

    def ppf(self, q: float) -> int | float | np.ndarray:
        """Return the smallest possible value of this random variable at which the probability mass to the left is greater than or equal to `q`.

        Args:
            q (float): The desired quantile.

        Returns:
            int | np.ndarray: The smallest value at which this distribution has mass >= `q` to the left of it.
        """
        mask = self.cdf_vals >= min(q, 0.9999)
        lowest_val_with_mass_over_q: np.ndarray = np.argmax(mask, axis=0)
        return (
            lowest_val_with_mass_over_q.item()
            if lowest_val_with_mass_over_q.size == 1
            else lowest_val_with_mass_over_q
        )

    def cdf(self, x: int | np.ndarray) -> float | np.ndarray:
        d = self.cdf_vals.shape[1]
        result: np.ndarray = self.cdf_vals[x, np.arange(d)]
        if result.size == 1:
            return result.item()
        return result

    def rvs(self, size: int | tuple) -> np.ndarray:
        U = np.random.uniform(size=size)
        draws = np.zeros((U.size, self.dimension))
        for i, u in enumerate(U.ravel()):
            draws[i] = self.ppf(u)
        draws = draws.reshape(size, -1)
        return draws

    @property
    def pmf_vals(self) -> np.ndarray:
        if self._pmf_vals is None:
            truncated_support = np.arange(self.max_value).reshape(-1, 1)
            self._pmf_vals = self._pmf(truncated_support)
            self._pmf_vals = self._pmf_vals / self._pmf_vals.sum(axis=0)
        return self._pmf_vals

    @property
    def cdf_vals(self) -> np.ndarray:
        if self._cdf_vals is None:
            self._cdf_vals = np.cumsum(self.pmf_vals, axis=0)
        return self._cdf_vals

    def expected_value(self) -> float | np.ndarray:
        raise NotImplementedError("Should be implemented by subclass.")

    def _pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this random variable takes on the value(s) x. Does not need to be normalized.

        Args:
            x (int | np.ndarray): The value(s) to compute the (possibly unnormalized) probability of.

        Returns:
            float | np.ndarray: The probability that this distribution takes on the value(s) x.
        """
        raise NotImplementedError("Should be implemented by subclass.")
