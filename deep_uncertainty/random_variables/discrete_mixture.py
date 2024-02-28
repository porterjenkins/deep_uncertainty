import numpy as np

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DiscreteMixture(DiscreteRandomVariable):
    def __init__(self, distributions: list[DiscreteRandomVariable], weights: np.ndarray):
        super().__init__(
            dimension=distributions[0].dimension, max_value=distributions[0].max_value
        )

        self.distributions = distributions
        self.weights = weights

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Calculate the probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray): The probability of x.
        """
        probabilities = []
        for distribution in self.distributions:
            probabilities.append(distribution.pmf(x))
        probabilities = np.stack(probabilities, axis=0)
        return np.average(probabilities, weights=self.weights, axis=0)

    def expected_value(self) -> float | np.ndarray:
        return np.dot(self.weights, np.array([x.expected_value() for x in self.distributions]))
