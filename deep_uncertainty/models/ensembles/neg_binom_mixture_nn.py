from __future__ import annotations

from typing import Iterable

import torch
from torch.distributions import NegativeBinomial

from deep_uncertainty.experiments.config import EnsembleConfig
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models.ensembles.base_discrete_mixture_nn import BaseDiscreteMixtureNN
from deep_uncertainty.random_variables import DiscreteMixture


class NegBinomMixtureNN(BaseDiscreteMixtureNN):
    """An ensemble of NegBinom Neural Nets that outputs a discrete probability distribution over [0, infinity] (truncated at `max_value`).

    The ensemble's predictions are combined as a mixture model with uniform weights.
    This model is not meant to be trained, and should strictly be used for prediction.
    """

    def __init__(self, members: Iterable[NegBinomNN], max_value: int = 2000):
        """Initialize a `NegBinomMixtureNN`.

        Args:
            members (Iterable[NegBinomNN]): The members of the ensemble.
            max_value (int, optional): The max value to output probabilities for. Defaults to 2000.
        """
        super(NegBinomMixtureNN, self).__init__(members=members, max_value=max_value)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output probability tensor over [0, `max_value`], with shape (N, `self.max_value`).
        """
        dists = []
        for member in self.members:
            mu, alpha = torch.split(member._predict_impl(x), [1, 1], dim=-1)
            mu = mu.flatten()
            alpha = alpha.flatten()

            # Convert to standard parametrization.
            eps = torch.tensor(1e-6, device=mu.device)
            var = mu + alpha * mu**2
            p = mu / torch.maximum(var, eps)
            failure_prob = torch.minimum(
                1 - p, 1 - eps
            )  # Torch docs lie and say this should be P(success).
            n = mu**2 / torch.maximum(var - mu, eps)

            dists.append(NegativeBinomial(total_count=n, probs=failure_prob))

        mixture = DiscreteMixture(distributions=dists, weights=torch.ones(len(dists)))
        return mixture.pmf(torch.arange(self.max_value).unsqueeze(1)).transpose(0, 1)

    @staticmethod
    def from_config(config: EnsembleConfig) -> NegBinomMixtureNN:
        """Construct a NegBinomMixtureNN from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            PoissonMixtureNN: The specified ensemble of NegBinomNN models.
        """
        # TODO: Support for changing max value?
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            member = NegBinomNN.load_from_checkpoint(path)
            members.append(member)
        return NegBinomMixtureNN(members=members)
