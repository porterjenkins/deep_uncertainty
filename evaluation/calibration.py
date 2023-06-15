import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from scipy.stats import norm

def get_z_score_for_given_probability(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the number of standard deviations to each side of the mean of a normal distribution that span an interval with probablity `p`.
    
    Args:
        p (float | ndarray): The probability of the desired interval.

    Returns:
        z (float | ndarray): The number of standard deviations defined by `p`.
    """
    z = norm.ppf(1 - (1 - p) / 2)
    return z

def plot_regression_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, sigma_pred: np.ndarray, num_bins: int = 9,
                                      ax: Optional[plt.Axes] = None, show: bool = True) -> None:
    """Given the targets and outputs of a regression model, along with its predictive uncertainty, plot a calibration curve.
    
    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        y_pred (ndarray, (n,)): The predicted values of the regression targets.
        sigma_pred (ndarray, (n,)): The predictive uncertainties (standard deviation) of the regression model.
        num_bins (int): Specifies how many probability thresholds to use for checking confidence interval calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    expected_proportions_in_confidence_intervals = np.linspace(0.1, 0.9, num=num_bins)
    pred_proportions_in_confidence_intervals = np.zeros(num_bins)

    for i, p in enumerate(expected_proportions_in_confidence_intervals):
        z = get_z_score_for_given_probability(p)
        confidence_interval_min, confidence_interval_max = y_pred - (z * sigma_pred), y_pred + (z * sigma_pred)
        target_is_in_confidence_interval = (y_true >= confidence_interval_min) & (y_true <= confidence_interval_max)
        pred_proportions_in_confidence_intervals[i] = target_is_in_confidence_interval.mean()
         
    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Predicted proportion of targets in p% confidence interval")
    ax.set_ylabel("Expected proportion of targets in p% confidence interval")
    ax.plot(expected_proportions_in_confidence_intervals, expected_proportions_in_confidence_intervals, linestyle='--', color='red', label='Perfectly calibrated')
    ax.plot(pred_proportions_in_confidence_intervals, expected_proportions_in_confidence_intervals, marker='o', linestyle='-', color='black', label='Model')
    ax.legend()
    plt.tight_layout()
    
    if show:
        plt.show()

def compute_calibration_score(y_true: np.ndarray, y_pred: np.ndarray, sigma_pred: np.ndarray, num_bins: int = 9) -> float:
    """Given the targets and outputs of a regression model, along with its predictive uncertainty, compute the calibration score of the model.

    The calibration score is defined as 1 minus the absolute value of the area between the calibration curve of a perfectly calibrated model (y = x)
    and the given model.
    
    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        y_pred (ndarray, (n,)): The predicted values of the regression targets.
        sigma_pred (ndarray, (n,)): The predictive uncertainties (standard deviation) of the regression model.
        num_bins (int): Specifies how many probability thresholds to use for checking confidence interval calibration. This corresponds to how
                        many points along the calibration curve will be computed when calculating the calibration score.
    """
    raise NotImplementedError("Method not implemented yet.")


if __name__ == "__main__":

    # Sanity check.

    x = np.linspace(0, 3, 1000)
    y_true = 3 * x
    y_pred = y_true + np.random.randn(1000)
    sigma_pred = np.ones_like(x)    # We have nearly perfect calibration here since our model knows its uncertainty is 1.

    plot_regression_calibration_curve(y_true, y_pred, sigma_pred)