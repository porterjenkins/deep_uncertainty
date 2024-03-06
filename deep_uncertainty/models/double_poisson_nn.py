from functools import partial

import torch
from torch import nn
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import BetaSchedulerType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.torchmetrics import ExpectedCalibrationError
from deep_uncertainty.evaluation.torchmetrics import YoungCalibration
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.backbones import ScalarMLP
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.training.beta_schedulers import CosineAnnealingBetaScheduler
from deep_uncertainty.training.beta_schedulers import LinearBetaScheduler
from deep_uncertainty.training.losses import double_poisson_nll


class DoublePoissonNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Double Poisson distribution over each regression target (conditioned on the input).

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None, optional): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
        beta_scheduler_type (BetaSchedulerType | None, optional): If specified, the type of beta scheduler to use for training loss (if applicable). Defaults to None.
        beta_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}. Defaults to None.
    """

    def __init__(
        self,
        backbone: Backbone,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
        beta_scheduler_type: BetaSchedulerType | None = None,
        beta_scheduler_kwargs: dict | None = None,
    ):
        if beta_scheduler_type == BetaSchedulerType.COSINE_ANNEALING:
            self.beta_scheduler = CosineAnnealingBetaScheduler(**beta_scheduler_kwargs)
        elif beta_scheduler_type == BetaSchedulerType.LINEAR:
            self.beta_scheduler = LinearBetaScheduler(**beta_scheduler_kwargs)
        else:
            self.beta_scheduler = None

        super(DoublePoissonNN, self).__init__(
            loss_fn=partial(
                double_poisson_nll,
                beta=(
                    self.beta_scheduler.current_value if self.beta_scheduler is not None else None
                ),
            ),
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.backbone = backbone
        self.head = nn.Linear(backbone.output_dim, 2)

        self.mean_calibration = YoungCalibration(
            param_list=["mu", "phi"],
            rv_class_type=DoublePoisson,
            mean_param_name="mu",
            is_scalar=isinstance(backbone, ScalarMLP),
        )
        self.ece = ExpectedCalibrationError(
            param_list=["mu", "phi"],
            rv_class_type=DoublePoisson,
        )
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.save_hyperparameters(ignore=["backbone"])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (logmu, logphi), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as (logmu, logphi)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, phi), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as (logmu, logphi)
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
        mu, phi = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        phi = phi.flatten()
        self.mse.update(mu, y.flatten())
        self.mae.update(mu, y.flatten())
        self.mape.update(mu, y.flatten())
        self.mean_calibration.update({"mu": mu, "phi": phi}, x, y.flatten())
        self.ece.update({"mu": mu, "phi": phi}, y.flatten())

    def on_train_epoch_end(self):
        if self.beta_scheduler is not None:
            self.beta_scheduler.step()
            self.loss_fn = partial(double_poisson_nll, beta=self.beta_scheduler.current_value)
