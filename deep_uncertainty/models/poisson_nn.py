from functools import partial
from typing import Type

import torch
from scipy.stats import poisson
from torch import nn
from torch.nn.functional import poisson_nll_loss
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.torchmetrics import ExpectedCalibrationError
from deep_uncertainty.evaluation.torchmetrics import YoungCalibration
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.backbones import ScalarMLP
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN


class PoissonNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Poisson distribution over each regression target (conditioned on the input).

    Args:
        backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        super(PoissonNN, self).__init__(
            loss_fn=partial(poisson_nll_loss, log_input=True),
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.backbone = backbone_type()
        self.head = nn.Linear(self.backbone.output_dim, 1)
        self.mean_calibration = YoungCalibration(
            ["mu"], poisson, mean_param_name="mu", is_scalar=isinstance(self.backbone, ScalarMLP)
        )
        self.ece = ExpectedCalibrationError(["mu"], poisson)
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as log(mu)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)  # Interpreted as log(mu)
        self.backbone.train()

        return torch.exp(y_hat)

    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "mape": self.mape,
            "mean_calibration": self.mean_calibration,
            "ece": self.ece,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        self.mse.update(y_hat.flatten(), y.flatten())
        self.mae.update(y_hat.flatten(), y.flatten())
        self.mape.update(y_hat.flatten(), y.flatten())
        self.mean_calibration.update({"mu": y_hat.flatten()}, x, y.flatten())
        self.ece.update({"mu": y_hat.flatten()}, y.flatten())
