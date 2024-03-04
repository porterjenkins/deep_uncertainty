from typing import Type

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric

from deep_uncertainty.evaluation.calibration import compute_young_calibration
from deep_uncertainty.evaluation.plotting import plot_posterior_predictive
from deep_uncertainty.evaluation.plotting import plot_regression_calibration_curve


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
        self.all_targets = torch.cat(self.y).long().flatten().detach().numpy()
        return compute_young_calibration(self.all_targets, self.posterior_predictive)

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
