from typing import Type

import numpy as np
import torch
from scipy.stats import nbinom
from torch import nn
<<<<<<< HEAD
=======
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
>>>>>>> f246b5f (Added Negative Binomial NN)
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.training.losses import neg_binom_nll


class NegBinomNN(DiscreteRegressionNN):
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

<<<<<<< HEAD
        self.discrete_ece = DiscreteExpectedCalibrationError(alpha=2)
        self.nll = DoublePoissonNLL()
        self.mp = MedianPrecision()

=======
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
>>>>>>> f246b5f (Added Negative Binomial NN)
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

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        dist = self._convert_output_to_dist(y_hat)
        return dist.mode

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
<<<<<<< HEAD
            "discrete_ece": self.discrete_ece,
            "nll": self.nll,
            "mp": self.mp,
=======
            "mse": self.mse,
            "mae": self.mae,
            "mape": self.mape,
>>>>>>> f246b5f (Added Negative Binomial NN)
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        dist = self._convert_output_to_dist(y_hat)
        mu, var = dist.mean, dist.variance
        precision = 1 / var
        preds = dist.mode
        probs = torch.exp(dist.log_prob(preds))
        targets = y.flatten()

        self.discrete_ece.update(preds=preds, probs=probs, targets=targets)
        self.nll.update(mu=mu, phi=mu / var, targets=targets)
        self.mp.update(precision)

    def _convert_output_to_dist(self, y_hat: torch.Tensor) -> torch.distributions.NegativeBinomial:
        """Convert a network output to the implied negative binomial distribution.

        Args:
            y_hat (torch.Tensor): Output from a `NegBinomNN` (nbinom parameters for the predicted distribution over y).

        Returns:
            torch.distributions.NegativeBinomial: The implied negative binomial distribution over y.
        """
        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        alpha = alpha.flatten()

        # Convert to scipy parametrization.
        var = mu + alpha * mu**2
<<<<<<< HEAD
        p = mu / torch.maximum(var, eps)
        failure_prob = torch.minimum(
            1 - p, 1 - eps
        )  # Torch docs lie and say this should be P(success).
        n = mu**2 / torch.maximum(var - mu, eps)
        dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
        return dist
=======
        p = mu / var
        n = mu**2 / (var - mu)

        probs = nbinom.pmf(
            np.arange(2000).reshape(-1, 1), n=n.detach().cpu().numpy(), p=p.detach().cpu().numpy()
        )
        mode = torch.tensor(np.argmax(probs, axis=0), device=self.mse.device)

        self.mse.update(mode, y.flatten())
        self.mae.update(mode, y.flatten())
        self.mape.update(mode, y.flatten())
>>>>>>> f246b5f (Added Negative Binomial NN)
