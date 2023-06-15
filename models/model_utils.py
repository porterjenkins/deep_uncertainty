import numpy as np
import torch
import matplotlib.pyplot as plt

def get_binom_p(mu: np.ndarray, n: np.ndarray):
    """
    Derive the binomial p parameter from the mean, and the n parameter
    :param mu: array of mu's
    :param n: array of n's
    :return: p: np.array
    """
    return mu / n

def get_binom_n(mu: np.array, sig2: np.array):
    """
    Get the binomial n parameter from mu and sig2
    :param mu: array of means
    :param sig2: array of variances
    :return: n: np.array
    """

    return mu/(1-(sig2/mu))



def get_1d_plot(X, y, model):

    x_for_pred = np.sort(X)
    x_for_pred = torch.Tensor(x_for_pred.reshape(-1, 1))

    model.eval()

    with torch.no_grad():
        y_predicted_tensor = model(x_for_pred)
        if isinstance(y_predicted_tensor, tuple):
            y_predicted_tensor = y_predicted_tensor[0]
        y_predicted = y_predicted_tensor.squeeze().numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label="Generated Data")
    plt.plot(x_for_pred, y_predicted, label="Predicted", color='r', linestyle='-', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Regression')
    plt.legend()
    plt.show()
