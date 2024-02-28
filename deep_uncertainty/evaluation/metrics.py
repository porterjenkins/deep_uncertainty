from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error
from torchmetrics import Metric

from deep_uncertainty.evaluation.calibration import compute_mean_calibration
from deep_uncertainty.evaluation.calibration import plot_regression_calibration_curve
from deep_uncertainty.evaluation.plots import plot_posterior_predictive


class MeanCalibration(Metric):
    """A custom `torchmetric` for computing mean calibration over multiple test batches in `lightning`.

    The mean calibration requires specifying a posterior predictive distribution over all test regression targets,
    and thus does not lend itself to the typical "collect, update, and aggregate" framework. This implementation
    accumulates distribution parameters output by a neural network and constructs the posterior predictive distribution
    at the final step. Invoking this class's `__call__` method will update its internal state with the specified parameters.

    Attributes:
        param_list (list): List of parameter names for the distribution being modeled. Should match the kwargs necessary to initialize `rv_class_type`.
        rv_class_type (Type): Type variable used to create an instance of the random variable whose parameters are output by the network, e.g. `scipy.stats.norm`.
        mean_param_name (str): For the specified RV, name of the parameter to treat as the mean, e.g. `"loc"`.
    """

    def __init__(self, param_list: list, rv_class_type: Type, mean_param_name: str, **kwargs):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.rv_class_type = rv_class_type
        self.mean_param_name = mean_param_name

        for param in param_list:
            self.add_state(param, default=[], dist_reduce_fx="cat")
        self.add_state("x", default=[], dist_reduce_fx="cat")
        self.add_state("y", default=[], dist_reduce_fx="cat")

    def update(self, params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):

        if list(params.keys()) != self.param_list:
            raise ValueError(
                f"""Must specify values for each param indicated in the `param_list` provided at `MeanCalibration` initialization.
                Got values for {list(params.keys())}, but was expecting values for {self.param_list}.
                """
            )

        for param_name, param_value in params.items():
            getattr(self, param_name).append(param_value)
        self.x.append(x)
        self.y.append(y)

    def compute(self) -> torch.Tensor:
        param_dict = {
            param_name: torch.cat(getattr(self, param_name)).flatten().detach().numpy()
            for param_name in self.param_list
        }
        self.posterior_predictive = self.rv_class_type(**param_dict)
        self.all_targets = torch.cat(self.y).flatten().detach().numpy()
        return compute_mean_calibration(self.all_targets, self.posterior_predictive)

    def plot(self) -> Figure:
        if not self._computed:
            raise ValueError("Must call `compute` before calling `plot`.")
        self.all_inputs = torch.cat(self.x).flatten().detach().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        if hasattr(self.posterior_predictive, self.mean_param_name):
            mean_preds = getattr(self.posterior_predictive, self.mean_param_name)
        else:
            mean_preds = self.posterior_predictive.kwds[self.mean_param_name]

        plot_regression_calibration_curve(
            self.all_targets, self.posterior_predictive, ax=axs[0], show=False
        )
        plot_posterior_predictive(
            x_test=self.all_inputs,
            y_test=self.all_targets,
            preds=mean_preds,
            upper=self.posterior_predictive.ppf(0.025),
            lower=self.posterior_predictive.ppf(0.975),
            show=False,
            ax=axs[1],
            title="Posterior Predictive",
        )
        return fig


def get_mse(y_true, y_hat):

    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.data.numpy()

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.data.numpy()

    return mean_squared_error(y_true=y_true, y_pred=y_hat)


def get_med_se(y_true, y_hat):
    return np.median((y_true - y_hat) ** 2)


def get_calibration(targets, upper, lower):
    if isinstance(targets, torch.Tensor):
        targets = targets.data.numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.data.numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.data.numpy()

    calib = np.mean(np.where((targets >= lower) & (targets <= upper), 1, 0))
    return calib
