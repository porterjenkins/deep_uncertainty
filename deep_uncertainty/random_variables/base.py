from abc import ABC as AbstractBaseClass
from abc import abstractmethod


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
