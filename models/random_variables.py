from typing import List, Union
from abc import (
    ABC as AbstractBaseClass,
    abstractmethod,
)
from scipy.stats import truncnorm, rv_discrete
from scipy.special import erf
import numpy as np


class BaseRandomVariable(AbstractBaseClass):
    """Base class for a random variable. Defines all properties and methods that a random variable class should implement."""

    @abstractmethod
    def nll(self, x):
        """Calculate the negative log likelihood of `x` for this random variable."""
        pass

    @abstractmethod
    def ppf(self, q):
        """Return the largest possible value of this random variable at which the probability mass to the left is less than or equal to `q`."""
        pass


class DiscreteRandomVariable(BaseRandomVariable):
    """Base class for a discrete random variable. Defines all properties and methods that a discrete random variable class should implement."""
    
    @abstractmethod
    def pmf(self, x):
        """Calculate the probability that this random variable takes on the value(s) `x`."""
        pass

    @abstractmethod
    def logpmf(self, x):
        """Calculate the log probability that this random variable takes on the value(s) `x`."""
        pass
    

class ContinuousRandomVariable(BaseRandomVariable):
    """Base class for a continuous random variable. Defines all properties and methods that a continuous random variable class should implement."""

    @abstractmethod
    def pdf(self, x):
        """Calculate the probability density of this random variable at the value(s) `x`."""
        pass

    @abstractmethod
    def logpdf(self, x):
        """Calculate the log probability density of this random variable at the value(s) `x`."""
        pass


class DiscreteConflation(DiscreteRandomVariable):
    """A conflation of discrete random variables, as described in https://www.researchgate.net/publication/1747728_Conflations_of_Probability_Distributions.
    
    Given discrete random variables X_1, X_2, ..., X_n, with pmfs g_{X_1}, g_{X_2}, ..., g_{X_n}, the conflation Q = X_1 & X_2 & ... & X_n has pmf given by

        g_Q(y) = (1 / Z) * g_{X_1}(y) * g_{X_2}(y) * ... * g_{X_n}(y)
        
    where Z is a normalizing constant so the masses sum to 1. This is a proper random variable with certain desirable properties for combining distributions.

    Args:
        random_variables (list(DiscreteRandomVariables)): A list of the discrete random variables to conflate (i.e. X_1, X_2, ..., X_n)
        support (np.ndarray): An array describing the support of the conflation (should be discrete). This should be the intersection of the support of X_1, ..., X_n.
    """
    def __init__(self, random_variables: List[Union[DiscreteRandomVariable, rv_discrete]], support: np.ndarray) -> None:
        self.random_variables = random_variables
        self.support = support
        self._fit()

    def _fit(self) -> None:
        """Precompute the pmf of this discrete conflation over the entire support."""
        running_product = np.ones_like(self.support, dtype=float)
        for rv in self.random_variables:
            running_product *= rv.pmf(self.support)

        self.unnormalized_mass = running_product
        self.Z = running_product.sum()
        self.mass = self.unnormalized_mass / self.Z

    def pmf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the probability that this conflation takes on the value(s) x.
        
        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Raises:
            ValueError: If x is not in the support of the conflation.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        if x not in self.support:
            raise ValueError(f"Provided value {x} is not in the support of this random variable.")
        
        probability = self.mass[np.where(self.support == x)]
        return probability
    
    def logpmf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the log probability that this conflation takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the log probability of.

        Raises:
            ValueError: If x is not in the support of the conflation.

        Returns:
            log_probability (float | np.ndarray): The log probability of x.
        """
        if x not in self.support:
            raise ValueError(f"Provided value {x} is not in the support of this random variable.")
        
        log_probability = np.log(self.pmf(x))
        return log_probability
    
    def nll(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the negative log likelihood of x for this conflation.
        
        Args:
            x (int | np.ndarray): The value(s) to compute the negative log likelihood of.

        Returns:
            nll (float | np.ndarray): The negative log likelihood of x.
        """
        nll = -self.logpmf(x)
        return nll
    
    def ppf(self, q: float) -> int:
        """Return the largest possible value of this conflation at which the probability mass to the left is less than or equal to `q`.
        
        Args:
            q (float): The desired quantile.
        """
        i = 0
        while self.mass[:i+1].sum() <= q:
            i += 1
        return int(self.support[i])
    

class DiscreteTruncatedNormal(DiscreteRandomVariable):
    """A discrete truncated normal random variable.
    
    This random variable has support [lower_bound, upper_bound] \\intersect \\mathbb{Z}.
    If X ~ DiscreteTruncatedNormal(lower_bound, upper_bound, mu, sigma), then let a = (lower_bound - mu) / sigma, b = (upper_bound - mu) / sigma, and
        let Y ~ TruncatedNormal(a, b, mu, sigma) have pdf f_Y. Then the pmf of X is given by P(X = x) = (1/Z) * f_Y(x) (where Z is a normalizing constant).

    Args:
        lower_bound (int): The lowest value the DiscreteTruncatedNormal random variable can take.
        upper_bound (int): The highest value the DiscreteTruncatedNormal random variable can take.
        mu (float, int): The mean of the truncated normal distribution that will be used to compute unnormalized densities.
        sigma (float, int): The standard deviation of the truncated normal distribution that will be used to compute unnormalized densities.
    """
    def __init__(self, lower_bound: int, upper_bound: int, mu: Union[float, int], sigma: Union[float, int]):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mu = mu
        self.sigma = sigma
        self.a, self.b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        self.base_rv = truncnorm(loc=mu, scale=sigma, a=self.a, b=self.b)
        self.support = np.arange(lower_bound, upper_bound)
        self.Z = self.base_rv.pdf(self.support).sum()

    def pmf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the probability that this random variable takes on the value(s) x.
        
        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        probability = self.base_rv.pdf(x) / self.Z
        return probability
    
    def logpmf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the log probability that this random variable takes on the value(s) x.
        
        Args:
            x (int | np.ndarray): The value(s) to compute the log probability of.

        Returns:
            log_probability (float | np.ndarray): The log probability of x.
        """
        log_probability = np.log(self.pmf(x))
        return log_probability
    
    def nll(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the negative log likelihood of x for this random variable.

        Note that the negative log likelihood here is not the true negative log likelihood of this random variable, but is proportional to it.
        
        Args:
            x (int | np.ndarray): The value(s) to compute the negative log likelihood of.

        Returns:
            nll (float | np.ndarray): The negative log likelihood of x.
        """
        nll = (
            0.5 * ((x - self.mu) / self.sigma)**2 +
            np.log(self.sigma) +
            np.log(erf((self.b - self.mu)/ (self.sigma * np.sqrt(2))) - erf((self.a - self.mu) / (self.sigma * np.sqrt(2))))
        )
        return nll
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Return the largest possible value of this random variable at which the probability mass to the left is less than or equal to `q`.
        
        Args:
            q (float): The desired quantile.
        """
        return np.floor(self.base_rv.ppf(q)).astype(int)