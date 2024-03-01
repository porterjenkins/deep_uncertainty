import torch
from torch.nn.functional import mse_loss
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import BackboneType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN


class MeanNN(BaseRegressionNN):
    """A neural network that fits to regression targets using mean squared error.

    Attributes:
        input_dim (int): Dimension of input data.
        backbone_type (BackboneType): The backbone type to use in the neural network, e.g. "mlp", "cnn", etc.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        input_dim: int,
        backbone_type: BackboneType,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        super(MeanNN, self).__init__(
            input_dim=input_dim,
            intermediate_dim=64,
            output_dim=1,
            backbone_type=backbone_type,
            loss_fn=mse_loss,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
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
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        return self._forward_impl(x)

    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {"mse": self.mse, "mae": self.mae, "mape": self.mape}

    def _update_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ) -> dict:
        self.mse.update(y_hat, y)
        self.mae.update(y_hat, y)
        self.mape.update(y_hat, y)
