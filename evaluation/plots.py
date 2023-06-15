import numpy as np
import torch
from matplotlib import pyplot as plt


def get_1d_sigma_plot(X, y, model):

    x_for_prediction = np.sort(X)
    # x_for_prediction = np.linspace(-6, 6, 100)
    x_for_prediction_tensor = torch.tensor(x_for_prediction, dtype=torch.float32).unsqueeze(1)

    model.eval()

    with torch.no_grad():
        mean_predicted_tensor, logvar_predicted_tensor = model(x_for_prediction_tensor)
        mean_predicted = mean_predicted_tensor.squeeze().numpy()
        std_predicted = np.sqrt(np.exp(logvar_predicted_tensor.squeeze().numpy()))

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.1, label="Generated Data")
    plt.plot(x_for_prediction, mean_predicted, label="Predicted Mean", color='r', linestyle='-', linewidth=2)
    plt.fill_between(x_for_prediction, mean_predicted - 2 * std_predicted, mean_predicted + 2 * std_predicted,
                     color='r', alpha=0.2, label="2 Std Dev")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian DNN Regression')
    plt.legend()
    plt.show()


def get_1d_mean_plot(X, y, model):

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
