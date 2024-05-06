from __future__ import annotations

from typing import Iterable

import torch
from torch.distributions import Poisson

from deep_uncertainty.experiments.config import EnsembleConfig
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.ensembles.base_discrete_mixture_nn import BaseDiscreteMixtureNN
from deep_uncertainty.random_variables import DiscreteMixture


class PoissonMixtureNN(BaseDiscreteMixtureNN):
    """An ensemble of Poisson Neural Nets that outputs a discrete probability distribution over [0, infinity] (truncated at `max_value`).

    The ensemble's predictions are combined as a mixture model with uniform weights.
    This model is not meant to be trained, and should strictly be used for prediction.
    """

    def __init__(self, members: Iterable[PoissonNN], max_value: int = 2000):
        """Initialize a `PoissonMixtureNN`.

        Args:
            members (Iterable[PoissonNN]): The members of the ensemble.
            max_value (int, optional): The max value to output probabilities for. Defaults to 2000.
        """
        super(PoissonMixtureNN, self).__init__(members=members, max_value=max_value)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output probability tensor over [0, `max_value`], with shape (N, `self.max_value`).
        """
        dists = []
        for member in self.members:
            mu = member._predict_impl(x)
            dists.append(Poisson(rate=mu.flatten()))

        mixture = DiscreteMixture(distributions=dists, weights=torch.ones(len(dists)))
        return mixture.pmf(torch.arange(self.max_value).unsqueeze(1)).transpose(0, 1)

    @staticmethod
    def from_config(config: EnsembleConfig) -> PoissonMixtureNN:
        """Construct a PoissonMixtureNN from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            PoissonMixtureNN: The specified ensemble of PoissonNN models.
        """
        # TODO: Support for changing max value?
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            member = PoissonNN.load_from_checkpoint(path)
            members.append(member)
        return PoissonMixtureNN(members=members)
