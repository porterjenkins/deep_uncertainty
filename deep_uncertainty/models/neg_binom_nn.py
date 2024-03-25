from typing import Type

import torch
from torch import nn
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.custom_torchmetrics import DiscreteExpectedCalibrationError
from deep_uncertainty.evaluation.custom_torchmetrics import DoublePoissonNLL
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.training.losses import neg_binom_nll


class NegBinomNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Negative Binomial distribution over each regression target (conditioned on the input).

    The mean-scale (mu, alpha) parametrization of the Negative Binomial is used for this network.

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
        beta_scheduler_type (BetaSchedulerType | None, optional): If specified, the type of beta scheduler to use for training loss (if applicable). Defaults to None.
        beta_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}. Defaults to None.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a NegBinomNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(NegBinomNN, self).__init__(
            loss_fn=neg_binom_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 2),
            nn.Softplus(),  # To ensure positivity of output params.
        )

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.discrete_ece = DiscreteExpectedCalibrationError(alpha=2)
        self.nll = DoublePoissonNLL()
        self.mp = MedianPrecision()

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()
        return y_hat

    def _test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "discrete_ece": self.discrete_ece,
            "nll": self.nll,
            "mp": self.mp,
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        alpha = alpha.flatten()

        # Convert to standard parametrization.
        eps = torch.tensor(1e-6, device=mu.device)
        var = mu + alpha * mu**2
        precision = 1 / var
        p = mu / torch.maximum(var, eps)
        failure_prob = torch.minimum(
            1 - p, 1 - eps
        )  # Torch docs lie and say this should be P(success).
        n = mu**2 / torch.maximum(var - mu, eps)
        dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
        preds = dist.mode
        probs = torch.exp(dist.log_prob(preds))

        targets = y.flatten()

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.discrete_ece.update(preds=preds, probs=probs, targets=targets)
        self.nll.update(mu=mu, phi=mu / var, targets=targets)
        self.mp.update(precision)
