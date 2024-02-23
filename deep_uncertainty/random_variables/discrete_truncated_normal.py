import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm


class DiscreteTruncatedNormal:
    """A discrete truncated normal random variable.

    This random variable has support [lower_bound, upper_bound] \\intersect \\mathbb{Z}.
    If X ~ DiscreteTruncatedNormal(lower_bound, upper_bound, mu, sigma), then let a = (lower_bound - mu) / sigma, b = (upper_bound - mu) / sigma, and
        let Y ~ TruncatedNormal(a, b, mu, sigma) have pdf f_Y. Then the pmf of X is given by P(X = x) = (1/Z) * f_Y(x) (where Z is a normalizing constant).

    Args:
        lower_bound (int): The lowest value the DiscreteTruncatedNormal random variable can take.
        upper_bound (int): The highest value the DiscreteTruncatedNormal random variable can take.
        mu (int | float | np.ndarray): The mean of the truncated normal distribution that will be used to compute unnormalized densities.
        sigma (int | float | np.ndarray): The standard deviation of the truncated normal distribution that will be used to compute unnormalized densities.
    """

    def __init__(
        self,
        lower_bound: int,
        upper_bound: int,
        mu: int | float | np.ndarray,
        sigma: int | float | np.ndarray,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mu = mu
        self.sigma = sigma
        self.a, self.b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        self.base_rv = truncnorm(loc=mu, scale=sigma, a=self.a, b=self.b)
        self.support = np.arange(lower_bound, upper_bound)
        self.Z = self.base_rv.pdf(self.support.reshape(-1, 1)).sum(axis=0)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        probability = self.base_rv.pdf(x) / self.Z
        return probability

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

        Note that the negative log likelihood here is not the true negative log likelihood of this random variable, but is proportional to it.

        Args:
            x (int | np.ndarray): The value(s) to compute the negative log likelihood of.

        Returns:
            nll (float | np.ndarray): The negative log likelihood of x.
        """
        nll = (
            0.5 * ((x - self.mu) / self.sigma) ** 2
            + np.log(self.sigma)
            + np.log(
                erf((self.b - self.mu) / (self.sigma * np.sqrt(2)))
                - erf((self.a - self.mu) / (self.sigma * np.sqrt(2)))
            )
        )
        return nll

    def ppf(self, q: float) -> int | np.ndarray:
        """Return the largest possible value of this random variable at which the probability mass to the left is less than or equal to `q`.

        Args:
            q (float): The desired quantile.

        Returns:
            int | np.ndarray: The largest value at which this distribution has mass <= `q` to the left of it.
        """
        return np.floor(self.base_rv.ppf(q)).astype(int)
