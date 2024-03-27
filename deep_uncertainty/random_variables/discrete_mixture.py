import numpy as np
import torch

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DiscreteMixture(DiscreteRandomVariable):
    """A mixture of discrete random variables, assumed to have support [0, infinity).

    This distribution's pmf is simply a weighted average of the underlying mixture members.
    """

    def __init__(
        self, distributions: list[DiscreteRandomVariable], weights: np.ndarray | torch.Tensor
    ):
        """Initialize a discrete mixture.

        Args:
            distributions (list[DiscreteRandomVariable]): The distributions to form a mixture with.
            weights (np.ndarray | torch.Tensor): The respective weights to place on each distribution. Automatically normalized to sum to 1.
        """
        if not all(dist.dimension == distributions[0].dimension for dist in distributions):
            raise ValueError("Distributions should all have same dimension if forming a mixture.")

        super().__init__(
            dimension=distributions[0].dimension,
            max_value=distributions[0].max_value,
            use_torch=distributions[0].use_torch,
            device=distributions[0].device,
        )
        self.distributions = distributions
        self.weights = (weights / weights.sum()).to(self.device)

    def _pmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        """Calculate the probability that this random variable takes on the value(s) x.

        Args:
            x (int | np.ndarray | torch.Tensor): The value(s) to compute the probability of.

        Returns:
            probability (float | np.ndarray | torch.Tensor): The probability of x.
        """
        probabilities = []
        for distribution in self.distributions:
            probabilities.append(distribution.pmf(x))
        if self.use_torch:
            probabilities = torch.stack(probabilities, dim=0)
            return (self.weights.view(-1, 1, 1) * probabilities).sum(dim=0)
        else:
            probabilities = np.stack(probabilities, axis=0)
            return np.average(probabilities, weights=self.weights, axis=0)

    def _expected_value(self) -> float | np.ndarray | torch.Tensor:
        if self.use_torch:
            return (
                self.weights.view(-1, 1)
                * torch.stack([x.expected_value for x in self.distributions])
            ).sum(dim=0)
        else:
            return np.dot(self.weights, np.array([x.expected_value for x in self.distributions]))
