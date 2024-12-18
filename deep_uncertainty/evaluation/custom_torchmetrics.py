from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric

from deep_uncertainty.evaluation.calibration import compute_continuous_ece


class ContinuousExpectedCalibrationError(Metric):
    """A custom `torchmetric` for computing the expected calibration error (for continuous regression) over multiple test batches in `lightning`.

    Attributes:
        param_list (list): List of parameter names for the distribution being modeled. Should match the kwargs necessary to initialize `rv_class_type`.
        rv_class_type (Type): Type variable used to create an instance of the random variable whose parameters are output by the network, e.g. `scipy.stats.norm`.
        num_bins (int): The number of bins to use for the ECE. Defaults to 30.
        weights (str, optional): Strategy for choosing the weights in the ECE sum. Must be either "uniform" or "frequency" (terms are weighted by the numerator of q_j). Defaults to "uniform".
        alpha (int, optional): Controls how severely we penalize the model for the distance between p_j and q_j. Defaults to 1 (error term is |p_j - q_j|^1).
    """

    def __init__(
        self,
        param_list: list,
        rv_class_type: Type,
        num_bins: int = 30,
        weights: str = "uniform",
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.rv_class_type = rv_class_type
        self.num_bins = num_bins
        self.weights = weights
        self.alpha = alpha

        for param in param_list:
            self.add_state(param, default=[], dist_reduce_fx="cat")

        self.add_state("y", default=[], dist_reduce_fx="cat")

    def update(self, params: dict[str, torch.Tensor], y: torch.Tensor):

        if list(params.keys()) != self.param_list:
            raise ValueError(
                f"""Must specify values for each param indicated in the `param_list` provided at `ExpectedCalibrationError` initialization.
                Got values for {list(params.keys())}, but was expecting values for {self.param_list}.
                """
            )

        for param_name, param_value in params.items():
            getattr(self, param_name).append(param_value)
        self.y.append(y)

    def compute(self) -> torch.Tensor:
        param_dict = {
            param_name: torch.cat(getattr(self, param_name)).flatten().detach().cpu().numpy()
            for param_name in self.param_list
        }
        self.posterior_predictive = self.rv_class_type(**param_dict)
        self.all_targets = torch.cat(self.y).long().flatten().detach().cpu().numpy()
        return torch.tensor(
            compute_continuous_ece(
                self.all_targets,
                self.posterior_predictive,
                self.num_bins,
                self.weights,
                self.alpha,
            ),
            device=self.device,
        )


class AverageNLL(Metric):
    """A custom `torchmetric` for computing the average NLL (negative log probability of the targets) over multiple test batches."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_state("all_target_probs", default=[], dist_reduce_fx="cat")

    def update(self, target_probs: torch.Tensor):
        """Update the state of this `AverageNLL` with the given target probabilities.

        Args:
            target_probs (torch.Tensor): Probabilities of the test targets (w.r.t a model's predicted conditional distribution) in a batch.
        """
        self.all_target_probs.append(target_probs)

    def compute(self) -> torch.Tensor:
        """Get the average NLL (negative log probability of all test targets).

        Returns:
            torch.Tensor: The average NLL.
        """
        all_target_probs = torch.cat(self.all_target_probs)
        eps = torch.tensor(1e-5, device=all_target_probs.device)
        nll = -torch.maximum(all_target_probs, eps).log().mean()
        return nll


class MedianPrecision(Metric):
    """A custom `torchmetric` for computing the median precision (1 / variance) of posterior predictive distributions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("all_precisions", default=[], dist_reduce_fx="cat")

    def update(self, precision: torch.Tensor):
        self.all_precisions.append(precision)

    def compute(self) -> torch.Tensor:
        all_precisions = torch.cat(self.all_precisions).flatten()
        return torch.median(all_precisions)

    def plot(self) -> Figure:
        precisions = torch.cat(self.all_precisions).flatten().detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1)

        upper = np.quantile(precisions, q=0.99)
        ax.hist(precisions[precisions <= upper], density=True)
        ax.set_title("Precision of Posterior Predictive")
        ax.set_xlabel("Precision")
        ax.set_ylabel("Density")

        return fig
