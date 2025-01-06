from __future__ import annotations

from typing import Generic
from typing import TypeVar

import torch
from lightning import LightningModule
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError

from deep_uncertainty.evaluation.custom_torchmetrics import AverageNLL
from deep_uncertainty.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.utils.configs import EnsembleConfig


T = TypeVar("T", bound=DiscreteRegressionNN)


class DeepRegressionEnsemble(LightningModule, Generic[T]):
    """Base class for deep regression ensembles. This model does not "train" and should strictly be used for prediction."""

    member_type: type[T] = None
    members: list[T]

    def __init__(self, members: list[T]):
        super().__init__()
        if not members:
            raise ValueError("The ensemble must have at least one member.")

        if not self.member_type:
            raise TypeError(
                "Subclasses of DeepEnsemble must define the `member_type` class attribute."
            )

        if not all(isinstance(member, self.member_type) for member in members):
            raise TypeError(f"All members must be instances of {self.member_type.__name__}.")

        self.members = members
        [member.eval() for member in self.members]

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.nll = AverageNLL()
        self.mp = MedianPrecision()
        self.crps = ContinuousRankedProbabilityScore(mode="discrete")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward (prediction) pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, ...).
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def _update_test_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Update the test metrics after a batch of predictions.

        Args:
            y_hat (torch.Tensor): Model output tensor on a batch of data, with shape (N, ...).
            y (torch.Tensor): Regression labels for the batch of data, with shape (N, 1).
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def from_config(cls, config: EnsembleConfig) -> DeepRegressionEnsemble[T]:
        """Construct an ensemble from a config. This is the primary way of building an ensemble."""
        checkpoint_paths = config.members
        members = [cls.member_type.load_from_checkpoint(path) for path in checkpoint_paths]
        return cls(members=members)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_impl(x)

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def test_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        self._update_test_metrics(y_hat, y.view(-1, 1).float())
        return y_hat

    def on_test_epoch_end(self):
        self.log("rmse", self.rmse.compute())
        self.log("mae", self.mae.compute())
        self.log("nll", self.nll.compute())
        self.log("mp", self.mp.compute())
        self.log("crps", self.crps.compute())
