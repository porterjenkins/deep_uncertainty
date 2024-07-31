from __future__ import annotations

import lightning as L
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.evaluation.custom_torchmetrics import AverageNLL
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models import MultiClassNN
from deep_uncertainty.utils.configs import EnsembleConfig


class MultiClassNNEnsemble(L.LightningModule):
    """An ensemble of MultiClass Neural Nets.

    The ensemble's predictions over a discrete set (meant to represent count regression targets) are uniformly averaged together.
    This model is not meant to be trained, and should strictly be used for prediction.
    """

    def __init__(self, members: list[MultiClassNN]):
        super(MultiClassNN, self).__init__()
        self.members = members
        [member.eval() for member in self.members]

        self.discrete_values = self.members[0].discrete_values
        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.nll = AverageNLL()
        self.mp = MedianPrecision()

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output class logits tensor, with shape (N, D). (D is the number of "classes" / regression targets the members predict over)
        """
        y_hat_vals = []
        for member in self.members:
            y_hat_vals.append(member._predict_impl(x))
        averaged_logits = torch.stack(y_hat_vals).mean(dim=0)

        return averaged_logits

    @property
    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "nll": self.nll,
            "mp": self.mp,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        preds = torch.tensor(
            data=[
                self.discrete_values[idx.item()] for idx in torch.argmax(y_hat, dim=-1).flatten()
            ],
            device=y_hat.device,
        )
        probs = torch.softmax(y_hat, dim=-1)
        targets = y.flatten()
        target_probs = torch.tensor(
            data=[probs[i][targets[i] - min(self.discrete_values)] for i in range(len(targets))],
            device=probs.device,
        )
        values = torch.tensor(self.discrete_values, device=probs.device, dtype=torch.float32)
        means = torch.matmul(probs, values)
        variances = ((values - means.view(-1, 1)) ** 2 * probs).sum(axis=1)
        precisions = 1 / variances

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.nll.update(target_probs=target_probs)
        self.mp.update(precisions)

    def test_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._predict_impl(x)
        self._update_test_metrics_batch(x, y_hat, y.flatten().long())
        return y_hat

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self._predict_impl(x)
        return y_hat

    def on_test_epoch_end(self):
        for name, metric_tracker in self._test_metrics_dict.items():
            self.log(name, metric_tracker.compute())

    @staticmethod
    def from_config(config: EnsembleConfig) -> MultiClassNNEnsemble:
        """Construct a MultiClassNNEnsemble from a config. This is the primary way of building an ensemble.

        Args:
            config (EnsembleConfig): Ensemble config object.

        Returns:
            MultiClassNNEnsemble: The specified ensemble of MultiClassNN models.
        """
        checkpoint_paths = config.members
        members = []
        for path in checkpoint_paths:
            member = MultiClassNN.load_from_checkpoint(path)
            members.append(member)
        return MultiClassNNEnsemble(members=members)
