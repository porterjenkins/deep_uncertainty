from typing import Type

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN


class MeanNN(BaseRegressionNN):
    """A neural network that fits to regression targets using mean squared error.

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
        super(MeanNN, self).__init__(
            loss_fn=mse_loss,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.backbone = backbone_type()
        self.head = nn.Linear(self.backbone.output_dim, 1)
        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()

        return y_hat

    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {"rmse": self.rmse, "mae": self.mae}

    def _update_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ) -> dict:
        preds = y_hat.flatten()
        targets = y.flatten()

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
