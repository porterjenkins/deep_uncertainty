from __future__ import annotations

from typing import Iterable

import lightning as L
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.evaluation.custom_torchmetrics import AverageNLL
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models import HomoscedasticGaussianNN
from deep_uncertainty.utils.configs import EnsembleConfig


class HomoscedasticGaussianMixtureNN(L.LightningModule):
    """An ensemble of Homoscedastic Gaussian Neural Nets that outputs the predictive mean and variance of the implied uniform mixture.

    See https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html for details.

    This model is not trained, and should strictly be used for prediction.

    Args:
        members (Iterable[GaussianNN]): The members of the ensemble.
    """

    def __init__(self, members: Iterable[HomoscedasticGaussianNN]):
        super(HomoscedasticGaussianMixtureNN, self).__init__()
        self.members = members
        [member.eval() for member in self.members]

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.nll = AverageNLL()
        self.mp = MedianPrecision()

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        mu_vals = []
        var_vals = []
        for member in self.members:
            y_hat = member._predict_impl(x)
            mu, var = torch.split(y_hat, [1, 1], dim=-1)
            mu_vals.append(mu)
            var_vals.append(var)
        mu_vals = torch.cat(mu_vals, dim=1)
        var_vals = torch.cat(var_vals, dim=1)

        mixture_mu = mu_vals.mean(dim=1)
        mixture_var = (var_vals + mu_vals**2).mean(dim=1) - mixture_mu**2

        return torch.stack([mixture_mu, mixture_var], dim=1)

    @property
    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "nll": self.nll,
            "mp": self.mp,
            "diversity": self.diversity,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, var = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        var = var.flatten()
        precision = 1 / var
        std = torch.sqrt(var)
        targets = y.flatten()

        preds = torch.round(mu)  # Since we have to predict counts.

        # We compute "probability" by normalizing density over the discrete counts.
        dist = torch.distributions.Normal(loc=mu, scale=std)
        target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.nll.update(target_probs=target_probs)
        self.mp.update(precision)

    def test_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._predict_impl(x)
        self._update_test_metrics_batch(x, y_hat, y.view(-1, 1).float())
        return y_hat

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self._predict_impl(x)
        return y_hat

    def on_test_epoch_end(self):
        for name, metric_tracker in self._test_metrics_dict.items():
            self.log(name, metric_tracker.compute())

    @staticmethod
    def from_config(config: EnsembleConfig) -> HomoscedasticGaussianMixtureNN:
        """Construct a HomoscedasticGaussianMixtureNN from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            GaussianMixtureNN: The specified ensemble of GaussianNN models.
        """
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            member = HomoscedasticGaussianNN.load_from_checkpoint(path)
            members.append(member)
        return HomoscedasticGaussianMixtureNN(members=members)
