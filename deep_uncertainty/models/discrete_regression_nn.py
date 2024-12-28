from typing import Callable
from typing import Type

import lightning as L
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.backbones import Backbone


class DiscreteRegressionNN(L.LightningModule):
    """Base class for discrete regression neural networks. Should not actually be used for prediction (needs to define `training_step` and whatnot).

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
        super(DiscreteRegressionNN, self).__init__()

        self.backbone = backbone_type(**backbone_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = loss_fn
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def configure_optimizers(self) -> dict:
        if self.optim_type == OptimizerType.ADAM:
            optim_class = torch.optim.Adam
        elif self.optim_type == OptimizerType.ADAM_W:
            optim_class = torch.optim.AdamW
        elif self.optim_type == OptimizerType.SGD:
            optim_class = torch.optim.SGD
        optimizer = optim_class(filter(lambda p: p.requires_grad, self.parameters()), **self.optim_kwargs)
        optim_dict = {"optimizer": optimizer}

        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING:
                lr_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
            lr_scheduler = lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            optim_dict["lr_scheduler"] = lr_scheduler

        num_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        print(f"Number of parameters tracked by the optimizer: {num_params}")

        return optim_dict

    def training_step(self, batch: tuple[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
        x, y = batch
        y_hat = self._forward_impl(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        with torch.no_grad():
            point_predictions = self._point_prediction(y_hat, training=True).flatten()
            self.train_rmse.update(point_predictions, y.flatten().float())
            self.train_mae.update(point_predictions, y.flatten().float())
            self.log("train_rmse", self.train_rmse, on_epoch=True)
            self.log("train_mae", self.train_mae, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._forward_impl(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Since we use _forward_impl, we specify training=True to get the proper transforms.
        point_predictions = self._point_prediction(y_hat, training=True).flatten()
        self.val_rmse.update(point_predictions, y.flatten().float())
        self.val_mae.update(point_predictions, y.flatten().float())
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)

        return loss

    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_hat = self._predict_impl(x)
        point_predictions = self._point_prediction(y_hat, training=False).flatten()
        self.test_rmse.update(point_predictions, y.flatten().float())
        self.test_mae.update(point_predictions, y.flatten().float())
        self._update_addl_test_metrics_batch(x, y_hat, y.view(-1, 1).float())

        self.log("test_rmse", self.test_rmse, on_epoch=True)
        self.log("test_mae", self.test_mae, on_epoch=True)
        for name, metric_tracker in self._addl_test_metrics_dict().items():
            self.log(name, metric_tracker, on_epoch=True)

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self._predict_impl(x)

        return y_hat

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

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        """Transform the network's output into a single discrete point prediction.

        This method will vary depending on the type of regression head (probabilistic vs. deterministic).
        For example, a gaussian regressor will return the `mean` portion of its output as its point prediction, rounded to the nearest integer.

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example.

        Returns:
            torch.Tensor: Point predictions for the true regression target, with shape (N, 1).
        """
        raise NotImplementedError("Should be implemented by subclass.")

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        """Return a dict with the metric trackers used by this model beyond the default rmse/mae."""
        raise NotImplementedError("Should be implemented by subclass.")

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        """Update additional test metric states (beyond default rmse/mae) given a batch of inputs/outputs/targets.

        Args:
            x (torch.Tensor): Model inputs.
            y_hat (torch.Tensor): Model predictions.
            y (torch.Tensor): Model regression targets.
        """
        raise NotImplementedError("Should be implemented by subclass.")
