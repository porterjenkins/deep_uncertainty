from __future__ import annotations

from typing import Iterable

import torch

from deep_uncertainty.models import NaturalGaussianNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.utils.configs import EnsembleConfig


class NaturalGaussianMixtureNN(GaussianMixtureNN):
    """An ensemble of naturally-parametrized Gaussian NNs that outputs the predictive mean and variance of the implied uniform mixture.

    See https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html for details.

    This model is not trained, and should strictly be used for prediction.

    Args:
        members (Iterable[NaturalGaussianNN]): The members of the ensemble.
    """

    def __init__(self, members: Iterable[NaturalGaussianNN]):
        super(NaturalGaussianMixtureNN, self).__init__(members)
        self.members = members
        [member.eval() for member in self.members]

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output mean and variance tensor, with shape (N, 2).

        If viewing outputs as (mu, var), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        mu_vals = []
        var_vals = []
        for member in self.members:
            eta_1, eta_2 = torch.split(member._predict_impl(x), [1, 1], dim=-1)
            mu = member._natural_to_mu(eta_1, eta_2)
            var = member._natural_to_var(eta_2)
            mu_vals.append(mu)
            var_vals.append(var)
        mu_vals = torch.cat(mu_vals, dim=1)
        var_vals = torch.cat(var_vals, dim=1)

        mixture_mu = mu_vals.mean(dim=1)
        mixture_var = (var_vals + mu_vals**2).mean(dim=1) - mixture_mu**2

        return torch.stack([mixture_mu, mixture_var], dim=1)

    @staticmethod
    def from_config(config: EnsembleConfig) -> NaturalGaussianMixtureNN:
        """Construct a NaturalGaussianMixtureNN from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            NaturalGaussianMixtureNN: The specified ensemble of NaturalGaussianNN models.
        """
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            member = NaturalGaussianNN.load_from_checkpoint(path)
            members.append(member)
        return NaturalGaussianMixtureNN(members=members)
