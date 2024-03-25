from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric

from deep_uncertainty.evaluation.calibration import compute_continuous_ece
from deep_uncertainty.evaluation.calibration import compute_discrete_ece
from deep_uncertainty.evaluation.calibration import compute_double_poisson_nll
from deep_uncertainty.evaluation.calibration import compute_young_calibration
from deep_uncertainty.evaluation.plotting import plot_posterior_predictive
from deep_uncertainty.evaluation.plotting import plot_regression_calibration_curve


class YoungCalibration(Metric):
    """A custom `torchmetric` for computing the Young calibration over multiple test batches in `lightning`.

    The Young calibration requires specifying a posterior predictive distribution over all test regression targets,
    and thus does not lend itself to the typical "collect, update, and aggregate" framework. This implementation
    accumulates distribution parameters output by a neural network and constructs the posterior predictive distribution
    at the final step. Invoking this class's `__call__` method will update its internal state with the specified parameters.

    Attributes:
        param_list (list): List of parameter names for the distribution being modeled. Should match the kwargs necessary to initialize `rv_class_type`.
        rv_class_type (Type): Type variable used to create an instance of the random variable whose parameters are output by the network, e.g. `scipy.stats.norm`.
        mean_param_name (str): For the specified RV, name of the parameter to treat as the mean, e.g. `"loc"`.
    """

    def __init__(
        self,
        param_list: list,
        rv_class_type: Type,
        mean_param_name: str,
        is_scalar: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.rv_class_type = rv_class_type
        self.mean_param_name = mean_param_name
        self.is_scalar = is_scalar

        for param in param_list:
            self.add_state(param, default=[], dist_reduce_fx="cat")
        if self.is_scalar:
            self.add_state("x", default=[], dist_reduce_fx="cat")
        self.add_state("y", default=[], dist_reduce_fx="cat")

    def update(self, params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):

        if list(params.keys()) != self.param_list:
            raise ValueError(
                f"""Must specify values for each param indicated in the `param_list` provided at `YoungCalibration` initialization.
                Got values for {list(params.keys())}, but was expecting values for {self.param_list}.
                """
            )

        for param_name, param_value in params.items():
            getattr(self, param_name).append(param_value)
        if self.is_scalar:
            self.x.append(x)
        self.y.append(y)

    def compute(self) -> torch.Tensor:
        param_dict = {
            param_name: torch.cat(getattr(self, param_name)).flatten().detach().cpu().numpy()
            for param_name in self.param_list
        }
        self.posterior_predictive = self.rv_class_type(**param_dict)
        self.all_targets = torch.cat(self.y).long().flatten().detach().cpu().numpy()
        return compute_young_calibration(self.all_targets, self.posterior_predictive)

    def plot(self) -> Figure:
        if not self._computed:
            raise ValueError("Must call `compute` before calling `plot`.")

        num_subplots = 2 if self.is_scalar else 1
        fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

        if hasattr(self.posterior_predictive, self.mean_param_name):
            mean_preds = getattr(self.posterior_predictive, self.mean_param_name)
        else:
            mean_preds = self.posterior_predictive.kwds[self.mean_param_name]

        if self.is_scalar:
            plot_regression_calibration_curve(
                self.all_targets, self.posterior_predictive, ax=axs[0], show=False
            )
            self.all_inputs = torch.cat(self.x).flatten().detach().cpu().numpy()
            plot_posterior_predictive(
                x=self.all_inputs,
                y=self.all_targets,
                mu=mean_preds,
                upper=self.posterior_predictive.ppf(0.025),
                lower=self.posterior_predictive.ppf(0.975),
                show=False,
                ax=axs[1],
                title="Posterior Predictive",
            )
        else:
            plot_regression_calibration_curve(
                self.all_targets, self.posterior_predictive, ax=axs, show=False
            )
        return fig


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
        return compute_continuous_ece(
            self.all_targets,
            self.posterior_predictive,
            self.num_bins,
            self.weights,
            self.alpha,
        )


class DiscreteExpectedCalibrationError(Metric):
    """A custom `torchmetric` for computing the expected calibration error (for count regression) over multiple test batches in `lightning`.

    Attributes:
        bin_strategy (str, optional): Strategy for choosing bin boundaries. Must be either "uniform" or "adaptive". Defaults to "adaptive" (same # of targets in each bin).
        alpha (int, optional): Controls how severely we penalize the model for the magnitude of a probability residual. Defaults to 1 (error term is |acc(B) - conf(B)|).
        num_bins (int): The number of bins to use. Defaults to 30.
    """

    def __init__(
        self,
        bin_strategy: str = "adaptive",
        alpha: float = 2.0,
        num_bins: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_strategy = bin_strategy
        self.alpha = alpha
        self.num_bins = num_bins

        self.add_state("all_preds", default=[], dist_reduce_fx="cat")
        self.add_state("all_probs", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor):
        self.all_preds.append(preds)
        self.all_probs.append(probs)
        self.all_targets.append(targets)

    def compute(self) -> torch.Tensor:
        all_preds = torch.cat(self.all_preds).long().flatten().detach().cpu().numpy()
        all_probs = torch.cat(self.all_probs).flatten().detach().cpu().numpy()
        all_targets = torch.cat(self.all_targets).long().flatten().detach().cpu().numpy()
        return compute_discrete_ece(
            targets=all_targets,
            preds=all_preds,
            probs=all_probs,
            bin_strategy=self.bin_strategy,
            alpha=self.alpha,
            num_bins=self.num_bins,
        )


class DoublePoissonNLL(Metric):
    """A custom `torchmetric` for computing the expected calibration error (for count regression) over multiple test batches in `lightning`."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.add_state("mu_vals", default=[], dist_reduce_fx="cat")
        self.add_state("phi_vals", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")

    def update(self, mu: torch.Tensor, phi: torch.Tensor, targets: torch.Tensor):
        self.mu_vals.append(mu)
        self.phi_vals.append(phi)
        self.all_targets.append(targets)

    def compute(self) -> torch.Tensor:
        mu_vals = torch.cat(self.mu_vals).long().flatten().detach().cpu().numpy()
        phi_vals = torch.cat(self.phi_vals).flatten().detach().cpu().numpy()
        all_targets = torch.cat(self.all_targets).long().flatten().detach().cpu().numpy()
        return compute_double_poisson_nll(
            y_true=all_targets,
            mu=mu_vals,
            phi=phi_vals,
        )


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
