import numpy as np
import torch
from sklearn.metrics import mean_squared_error



def get_mse(y_true, y_hat):

    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.data.numpy()

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.data.numpy()

    return mean_squared_error(y_true=y_true, y_pred=y_hat)


def get_med_se(y_true, y_hat):
    return np.median((y_true - y_hat)**2)


def get_calibration(targets, upper, lower):
    if isinstance(targets, torch.Tensor):
        targets = targets.data.numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.data.numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.data.numpy()

    calib = np.mean(
        np.where(
            (targets >= lower) & (targets <= upper),
            1,
            0
        )
    )
    return calib