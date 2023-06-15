import torch
from sklearn.metrics import mean_squared_error



def get_mse(y_true, y_hat):

    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.data.numpy()

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.data.numpy()

    return mean_squared_error(y_true=y_true, y_pred=y_hat)