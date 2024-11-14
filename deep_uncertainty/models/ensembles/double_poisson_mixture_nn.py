from __future__ import annotations

from typing import Iterable

import torch

from deep_uncertainty.models import DoublePoissonHomoscedasticNN
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models.ensembles.base_discrete_mixture_nn import BaseDiscreteMixtureNN
from deep_uncertainty.random_variables import DiscreteMixture
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.configs import EnsembleConfig


class DoublePoissonMixtureNN(BaseDiscreteMixtureNN):
    """An ensemble of Double Poisson Neural Nets that outputs a discrete probability distribution over [0, infinity] (truncated at `max_value`).

    The ensemble's predictions are combined as a mixture model with uniform weights.
    This model is not meant to be trained, and should strictly be used for prediction.
    """

    def __init__(
        self,
        members: Iterable[DoublePoissonNN | DoublePoissonHomoscedasticNN],
        max_value: int = 2000,
    ):
        """Initialize a `DoublePoissonMixtureNN`.

        Args:
            members (Iterable[DoublePoissonNN | DoublePoissonHomoscedasticNN]): The members of the ensemble.
            max_value (int, optional): The max value to output probabilities for. Defaults to 2000.
        """
        super(DoublePoissonMixtureNN, self).__init__(members=members, max_value=max_value)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output probability tensor over [0, `max_value`], with shape (N, `self.max_value`).
        """
        dists = []
        for member in self.members:
            mu, phi = torch.split(member._predict_impl(x), [1, 1], dim=-1)
            dists.append(DoublePoisson(mu=mu.flatten(), phi=phi.flatten()))

        mixture = DiscreteMixture(distributions=dists, weights=torch.ones(len(dists)))
        return mixture.pmf(torch.arange(self.max_value).unsqueeze(1)).transpose(0, 1)

    @staticmethod
    def from_config(config: EnsembleConfig) -> DoublePoissonMixtureNN:
        """Construct a DoublePoissonMixtureNN from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            DoublePoissonMixtureNN: The specified ensemble of DoublePoissonNN models.
        """
        # TODO: Support for changing max value?
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            try:
                member = DoublePoissonNN.load_from_checkpoint(path)
            except RuntimeError:
                member = DoublePoissonHomoscedasticNN.load_from_checkpoint(path)
            members.append(member)
        return DoublePoissonMixtureNN(members=members)
