import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import poisson
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from deep_uncertainty.evaluation.calibration import compute_mean_calibration
from deep_uncertainty.evaluation.calibration import plot_regression_calibration_curve
from deep_uncertainty.evaluation.metrics import get_calibration
from deep_uncertainty.evaluation.metrics import get_mse
from deep_uncertainty.evaluation.plots import get_sigma_plot_from_test
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.model_utils import get_mean_preds_and_targets_DPR


class DoublePoissonNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DoublePoissonNN, self).__init__()

        # Common layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # (1, 5)

        # Layer for beta predictions
        self.beta_layer = nn.Linear(hidden_size, 1)  # (5, 1)

        # Layer for alpha predictions
        self.alpha_layer = nn.Linear(hidden_size, 1)  # (5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        beta = self.beta_layer(x)
        alpha = torch.exp(
            self.alpha_layer(x)
        )  # Ensure alpha is positive since it's a scale parameter
        # lamdba (training: logits)
        return beta, alpha


def neg_log_factorial(value):
    # The input value is expected to be a tensor
    # Use lgamma function to compute log factorial
    log_fact = torch.lgamma(value + 1)
    return -log_fact


def double_poisson_loss(y_pred, y_true, x):

    # fix alpha (=1), to see: beta
    # fix ...
    # domain of paramters, log of alpha/beta (focus on the domain range of parameters)

    beta, gamma = y_pred  # (32, 1)

    # Ensure alpha is positive to prevent invalid values in the log likelihood
    # beta = beta.squeeze(-1)
    # alpha = alpha.squeeze(-1).clamp(min=1e-6)

    # Compute lambda_i from beta and x
    # ---- x.T: (1, 32), and beta: (32, 1)

    update_term = None

    for i in range(len(beta)):  # index
        lambda_i = torch.exp(x[i].T * beta.clamp(max=10))  # (1) * (32, 1) = (32, 1)
        # lambda_i = torch.exp(torch.mul(x[i].T , beta.clamp(max=10)))# (1, 32) * (32, 1) = (1, 1)
        term1 = -torch.log(
            (1 + ((1 - gamma) / (12 * gamma * lambda_i)) * (1 + 1 / (gamma * lambda_i))).clamp(
                max=10, min=1e-6
            )
        )  # (32, 1)
        term2 = -y_true[i]  # ([1])
        term3 = y_true[i] * torch.log(y_true[i])  # ([1])
        term4 = neg_log_factorial(y_true[i])  # ([1])
        term5 = gamma * y_true[i] * (1 + torch.log(lambda_i / y_true[i]))  # (32, 1)
        term_last = 0.5 * torch.log(gamma) - gamma * lambda_i  # (32, 1)

        final_i = term1 + term2 + term3 + term4 + term5 + term_last
        if update_term is None:
            update_term = final_i
        else:
            update_term += final_i
        update_term = -1 * update_term

    # lambda_i = torch.exp(x.T * beta.clamp(max=10)) # lambda_i： (1, 1)
    # safe_log_lambda = torch.log(lambda_i + 1e-6)

    # # y_true = y_true.squeeze(-1).clamp(min=1e-6)
    # # y_true = torch.clamp(y_true, min=1e-3)

    # safe_division = (1 + 1 / (12 * y_true * lambda_i + 1e-6))  # Prevent division by zero

    # # Log likelihood computation as per the provided image
    # # log_likelihood = -torch.log(1 + 1 / (12 * y_true * lambda_i)) - \
    # #                  torch.lgamma(y_true + 1) + y_true * torch.log(lambda_i) - lambda_i
    # log_likelihood = -torch.log(safe_division) - \
    #                  torch.lgamma(y_true + 1) + y_true * safe_log_lambda - lambda_i

    # safe_log_alpha = torch.log(alpha + 1e-6)  # Prevent log(0)
    # # Adding the alpha terms from the image to the log likelihood
    # log_likelihood += y_true * safe_log_alpha + (1 + y_true * safe_log_lambda) / alpha

    return torch.mean(update_term).clamp(max=10)


def main(config: dict):

    X_train = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_x_train.txt")
    )
    X_val = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_x_val.txt")
    )
    X_test = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_x_test.txt")
    )
    y_train = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_y_train.txt")
    )
    y_val = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_y_val.txt")
    )
    y_test = np.loadtxt(
        fname=os.path.join(config["dataset"]["dir"], config["dataset"]["name"] + "_y_test.txt")
    )

    train_dataset = TensorDataset(
        torch.Tensor(X_train.reshape(-1, 1)), torch.Tensor(y_train.reshape(-1, 1))
    )

    val_dataset = TensorDataset(
        torch.Tensor(X_val.reshape(-1, 1)), torch.Tensor(y_val.reshape(-1, 1))
    )

    test_dataset = TensorDataset(
        torch.Tensor(X_test.reshape(-1, 1)), torch.Tensor(y_test.reshape(-1, 1))
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Check for CUDA availability and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.
    model = DoublePoissonNN(input_size=1, hidden_size=5)  # input: (32, 1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    trn_losses = []
    val_losses = []

    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # input: (32, 1), output: (32, 1)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            beta, alpha = model(X_batch)  # beta: (32, 1), alpha: (32, 1)
            loss = double_poisson_loss((beta, alpha), y_batch, X_batch)
            loss.backward()
            optimizer.step()
            # train_loss += loss.item() * X_batch.size(0)
            train_loss += loss.item()
            trn_losses.append(loss.item())

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                beta, alpha = model(X_batch)
                loss = double_poisson_loss((beta, alpha), y_batch, X_batch)
                val_loss += loss.item()
                val_losses.append(loss.item())

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    test_preds, test_targets = get_mean_preds_and_targets_DPR(test_loader, model, device)

    prob = poisson(test_preds.data.numpy().flatten())
    lower = prob.ppf(0.025)
    upper = prob.ppf(0.975)

    test_mse = get_mse(test_targets, test_preds)
    print("Test MSE: {:.4f}".format(test_mse))

    test_calib = get_calibration(test_targets.flatten(), upper.flatten(), lower.flatten())
    print("95% Test Calib: {:.4f}".format(test_calib))

    mean_calib = compute_mean_calibration(test_targets.data.numpy().flatten(), prob)
    print("Mean Calib: {:.4f}".format(mean_calib))

    plt.plot(np.arange(num_epochs), trn_losses, label="TRAIN")
    plt.plot(np.arange(num_epochs), val_losses, label="VAL")
    plt.legend()
    plt.show()

    get_sigma_plot_from_test(
        X_test, y_test, test_preds, upper=upper.flatten(), lower=lower.flatten()
    )
    plot_regression_calibration_curve(test_targets.data.numpy().flatten(), prob, num_bins=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path = "./deep_uncertainty/toy/toy_exp_train_config.yaml"
    relative_path = path
    parser.add_argument(
        "--trn-cfg", type=str, default=relative_path, help="Path to data config file"
    )
    args = parser.parse_args()

    cfg = get_yaml(args.trn_cfg)

    main(config=cfg)
