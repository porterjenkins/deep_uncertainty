from functools import partial
from typing import Type

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.custom_torchmetrics import AverageNLL
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN


def is_contiguous(nums: list[int]):
    if len(nums) < 2:
        return True
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1] + 1:
            return False
    return True


def shifted_cross_entropy(input: torch.Tensor, target: torch.Tensor, shift: int) -> torch.Tensor:
    return cross_entropy(input, target.long().flatten() - shift)


class MultiClassNN(DiscreteRegressionNN):
    """A neural network that approximates discrete regression on a finite set via multiclass learning through cross-entropy loss."""

    def __init__(
        self,
        discrete_values: list[int],
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a MultiClassNN.

        Args:
            discrete_values (list[int]): The discrete regression values this network will output classification logits for. Must be contiguous (incrementing by one each time).
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        if not is_contiguous(discrete_values):
            raise ValueError("Must supply contiguous list of discrete values.")
        self.discrete_values = discrete_values

        super(MultiClassNN, self).__init__(
            loss_fn=partial(shifted_cross_entropy, shift=min(discrete_values)),
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.num_discrete_values = len(discrete_values)
        self.head = nn.Linear(self.backbone.output_dim, self.num_discrete_values)
        self.nll = AverageNLL()
        self.mp = MedianPrecision()

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output class logits tensor, with shape (N, `self.num_discrete_values`).
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output class logits tensor, with shape (N, `self.num_discrete_values`).
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()

        return y_hat

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        return torch.tensor(
            data=[
                self.discrete_values[idx.item()] for idx in torch.argmax(y_hat, dim=-1).flatten()
            ],
            device=y_hat.device,
        )

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "nll": self.nll,
            "mp": self.mp,
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        probs = torch.softmax(y_hat, dim=-1)
        targets = y.flatten().long()
        target_probs = torch.tensor(
            data=[probs[i][targets[i] - min(self.discrete_values)] for i in range(len(targets))],
            device=probs.device,
        )
        values = torch.tensor(self.discrete_values, device=probs.device, dtype=torch.float32)
        means = torch.matmul(probs, values)
        variances = ((values - means.view(-1, 1)) ** 2 * probs).sum(axis=1)

        precisions = 1 / variances

        self.nll.update(target_probs)
        self.mp.update(precisions)
