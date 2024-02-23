import numpy as np
from scipy.stats import rv_discrete

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DiscreteConflation:
    """A conflation of discrete random variables, as described in https://www.researchgate.net/publication/1747728_Conflations_of_Probability_Distributions.

    Given discrete random variables X_1, X_2, ..., X_n, with pmfs g_{X_1}, g_{X_2}, ..., g_{X_n}, the conflation Q = X_1 & X_2 & ... & X_n has pmf given by

        g_Q(y) = (1 / Z) * g_{X_1}(y) * g_{X_2}(y) * ... * g_{X_n}(y)

    where Z is a normalizing constant so the masses sum to 1. This is a proper random variable with certain desirable properties for combining distributions.

    Args:
        random_variables (list(DiscreteRandomVariables)): A list of the discrete random variables to conflate (i.e. X_1, X_2, ..., X_n)
        support (np.ndarray): An array describing the support of the conflation (should be discrete). This should be the intersection of the support of X_1, ..., X_n.
    """

    def __init__(
        self,
        random_variables: list[DiscreteRandomVariable | rv_discrete],
        support: np.ndarray,
    ) -> None:
        self.random_variables = random_variables
        self.support = support
        self._fit()

    def _fit(self) -> None:
        """Precompute the pmf of this discrete conflation over the entire support."""
        running_product = np.ones_like(self.support, dtype=float)
        for rv in self.random_variables:
            running_product *= rv.pmf(self.support).squeeze()

        self.unnormalized_mass = running_product
        self.Z = running_product.sum()
        self.mass = self.unnormalized_mass / self.Z

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this conflation takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Raises:
            ValueError: If x is not in the support of the conflation.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        if not np.all(np.in1d(x, self.support)):
            raise ValueError(f"Provided value(s) {x} not in the support of this random variable.")

        if isinstance(x, np.ndarray):
            indices = [np.where(self.support == val)[0][0] for val in x if val in self.support]
            probability = self.mass[indices]

        else:
            probability = self.mass[np.where(self.support == x)][0]

        return probability

    def logpmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the log probability that this conflation takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the log probability of.

        Raises:
            ValueError: If x is not in the support of the conflation.

        Returns:
            log_probability (float | np.ndarray): The log probability of x.
        """
        if not np.all(np.in1d(x, self.support)):
            raise ValueError(f"Provided value(s) {x} not in the support of this random variable.")

        log_probability = np.log(self.pmf(x))
        return log_probability

    def nll(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the negative log likelihood of x for this conflation.

        Args:
            x (int | np.ndarray): The value(s) to compute the negative log likelihood of.

        Returns:
            nll (float | np.ndarray): The negative log likelihood of x.
        """
        nll = -self.logpmf(x)
        return nll

    def ppf(self, q: float) -> int | np.ndarray:
        """Return the smallest possible value of this conflation at which the probability mass to the left is greater than or equal to `q`.

        Args:
            q (float): The desired quantile.

        Returns:
            int | np.ndarray: The smallest value at which this distribution has mass >= `q` to the left of it.
        """
        mass = self.pmf(self.support.reshape(-1, 1)).reshape(-1, 1)
        mass = mass / mass.sum(axis=0)  # Sometimes, the resultant mass isn't entirely normalized.
        mask = np.cumsum(mass, axis=0) >= min(q, 0.9999)
        indices = np.argmax(mask, axis=0)
        values = self.support[indices]
        return values.item() if values.size == 1 else values
