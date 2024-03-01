from functools import partial

import torch
from scipy.stats import norm
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import BackboneType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.torchmetrics import MeanCalibration
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.training.losses import gaussian_nll


class GaussianNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Gaussian distribution over each regression target (conditioned on the input).

    Attributes:
        input_dim (int): Dimension of input data.
        backbone_type (BackboneType): The backbone type to use in the neural network, e.g. "mlp", "cnn", etc.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        beta (float | None, optional): Beta parameter to use in Gaussian NLL loss. If specified, must be in [0, 1]. Defaults to None.
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

        super(GaussianNN, self).__init__(
            input_dim=input_dim,
            intermediate_dim=64,
            output_dim=2,
            backbone_type=backbone_type,
            loss_fn=partial(gaussian_nll, beta=beta),
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.mean_calibration = MeanCalibration(
            param_list=["loc", "scale"], rv_class_type=norm, mean_param_name="loc"
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

        If viewing outputs as (mu, logvar), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as (mu, logvar)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, var), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        y_hat = self._forward_impl(x)

        # Apply torch.exp to the logvar dimension.
        output_shape = y_hat.shape
        reshaped = y_hat.view(-1, 2)
        y_hat = torch.stack([reshaped[:, 0], torch.exp(reshaped[:, 1])], dim=1).view(*output_shape)

        return y_hat

    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "mape": self.mape,
            "mean_calibration": self.mean_calibration,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, var = torch.split(y_hat, [1, 1], dim=-1)
        self.mse.update(mu, y)
        self.mae.update(mu, y)
        self.mape.update(mu, y)

        std = torch.sqrt(var)
        self.mean_calibration.update({"loc": mu, "scale": std}, x, y)
