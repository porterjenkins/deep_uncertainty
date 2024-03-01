from functools import partial

import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import BackboneType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.torchmetrics import MeanCalibration
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.training.losses import double_poisson_nll


class DoublePoissonNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Double Poisson distribution over each regression target (conditioned on the input).

    Attributes:
        input_dim (int): Dimension of input data.
        backbone_type (BackboneType): The backbone type to use in the neural network, e.g. "mlp", "cnn", etc.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None, optional): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
        beta (float | None, optional): Beta parameter to use in Double Poisson NLL loss. If specified, must be in [0, 1]. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        backbone_type: BackboneType,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
        beta: float | None = None,
    ):
        if beta is not None and (beta < 0 or beta > 1):
            raise ValueError(f"If beta is specified, it must be in [0, 1]. Got value {beta}")

        super(DoublePoissonNN, self).__init__(
            input_dim=input_dim,
            intermediate_dim=64,
            output_dim=2,
            backbone_type=backbone_type,
            loss_fn=partial(double_poisson_nll, beta=beta),
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.mean_calibration = MeanCalibration(
            param_list=["mu", "phi"], rv_class_type=DoublePoisson, mean_param_name="mu"
        )
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

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
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

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
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, phi = torch.split(y_hat, [1, 1], dim=-1)
        self.mse.update(mu, y)
        self.mae.update(mu, y)
        self.mape.update(mu, y)
        self.mean_calibration.update({"mu": mu, "phi": phi}, x, y)
