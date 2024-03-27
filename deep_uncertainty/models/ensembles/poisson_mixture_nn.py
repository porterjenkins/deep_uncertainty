from __future__ import annotations

from typing import Iterable

import lightning as L
import torch
from matplotlib.figure import Figure
from torch.distributions import Poisson
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.evaluation.custom_torchmetrics import DiscreteExpectedCalibrationError
from deep_uncertainty.evaluation.custom_torchmetrics import DoublePoissonNLL
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.experiments.config import EnsembleConfig
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.random_variables import DiscreteMixture


class PoissonMixtureNN(L.LightningModule):
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
        super(PoissonMixtureNN, self).__init__()
        self.members = members
        self.max_value = max_value
        [member.eval() for member in self.members]

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.discrete_ece = DiscreteExpectedCalibrationError(alpha=2)
        self.nll = DoublePoissonNLL()
        self.mp = MedianPrecision()

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

    @property
    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "discrete_ece": self.discrete_ece,
            "nll": self.nll,
            "mp": self.mp,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):

        preds = y_hat.argmax(dim=1)
        probs = y_hat[torch.arange(y_hat.size(0), device=self.device), preds]
        targets = y.flatten()

        support = torch.arange(self.max_value, device=self.device)
        mu = (y_hat * support).sum(dim=1)
        var = (y_hat * (support.unsqueeze(1) - mu).transpose(0, 1) ** 2).sum(dim=1)
        precision = 1 / var

        self.discrete_ece.update(
            preds=preds,
            probs=probs,
            targets=targets,
        )
        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.nll.update(mu=mu, phi=mu / var, targets=targets)
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
            if name in {"mean_calibration", "mp"}:
                fig: Figure = metric_tracker.plot()
                if self.logger is not None:
                    root = self.logger.log_dir or "."
                else:
                    root = "."
                fig.savefig(root + f"/{name}_plot.png")

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
