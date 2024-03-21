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
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.base_regression_nn import BaseRegressionNN
from deep_uncertainty.training.losses import neg_binom_nll


class NegBinomNN(BaseRegressionNN):
    """A neural network that learns the parameters of a Negative Binomial distribution over each regression target (conditioned on the input).

    The mean-scale (mu, alpha) parametrization of the Negative Binomial is used for this network.

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
        super(NegBinomNN, self).__init__(
            loss_fn=neg_binom_nll,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.backbone = backbone_type()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 2),
            nn.Softplus(),  # To ensure positivity of output params.
        )

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.discrete_ece = DiscreteExpectedCalibrationError(alpha=2)
        self.nll = DoublePoissonNLL()

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
        }

    def _update_test_metrics_batch(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        alpha = alpha.flatten()

        # Convert to standard parametrization.
        var = mu + alpha * mu**2
        p = mu / var
        failure_prob = 1 - p  # Torch docs lie and say this should be P(success).
        n = mu**2 / (var - mu)
        dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
        preds = dist.mode
        probs = torch.exp(dist.log_prob(preds))

        targets = y.flatten()

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.discrete_ece.update(preds=preds, probs=probs, targets=targets)
        self.nll.update(mu=mu, phi=mu / var, targets=targets)
