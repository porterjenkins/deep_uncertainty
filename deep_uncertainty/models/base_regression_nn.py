from typing import Callable
from typing import Type

import lightning as L
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.backbones import Backbone


class BaseRegressionNN(L.LightningModule):
    """Base class for regression neural networks. Should not actually be used for prediction (needs to define `training_step` and whatnot).

    Attributes:
        backbone (Backbone): The backbone to use for feature extraction (before applying the regression head).
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a regression NN.

        Args:
            loss_fn (Callable): The loss function to use for training this NN.
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(BaseRegressionNN, self).__init__()

        self.backbone = backbone_type(**backbone_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = loss_fn

    def configure_optimizers(self) -> dict:
        if self.optim_type == OptimizerType.ADAM:
            optim_class = torch.optim.Adam
        elif self.optim_type == OptimizerType.ADAM_W:
            optim_class = torch.optim.AdamW
        elif self.optim_type == OptimizerType.SGD:
            optim_class = torch.optim.SGD
        optimizer = optim_class(self.parameters(), **self.optim_kwargs)
        optim_dict = {"optimizer": optimizer}

        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING:
                lr_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
            lr_scheduler = lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            optim_dict["lr_scheduler"] = lr_scheduler

        return optim_dict

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._forward_impl(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._forward_impl(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

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
        for name, metric_tracker in self._test_metrics_dict().items():
            self.log(name, metric_tracker.compute())
            if name in {"mean_calibration", "mp"}:
                fig: Figure = metric_tracker.plot()
                if self.logger is not None:
                    root = self.logger.log_dir or "."
                else:
                    root = "."
                fig.savefig(root + f"/{name}_plot.png")

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        raise NotImplementedError("Should be implemented by subclass.")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        This method will often differ from `_forward_impl` in cases where
        the output used for training is in log (or some other modified)
        space for numerical convenience.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        raise NotImplementedError("Should be implemented by subclass.")

    def _test_metrics_dict(self) -> dict[str, Metric]:
        """Return a dict with the metric trackers used by this model."""
        raise NotImplementedError("Should be implemented by subclass.")

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        """Update test metric states given a batch of outputs/targets.

        Args:
            x (torch.Tensor): Model inputs.
            y_hat (torch.Tensor): Model predictions.
            y (torch.Tensor): Model regression targets.
        """
        raise NotImplementedError("Should be implemented by subclass.")
