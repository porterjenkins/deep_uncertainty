import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from deep_uncertainty.evaluation.old.evals import evaluate_model_mse
from deep_uncertainty.evaluation.old.metrics import get_mse
from deep_uncertainty.evaluation.old.plots import get_1d_mean_plot
from deep_uncertainty.models.old.regressors import OldRegressionNN
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.model_utils import get_mean_preds_and_targets
from deep_uncertainty.utils.model_utils import train_regression_nn


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

    # Instantiate and train the network
    model = OldRegressionNN(log_output=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Train the network
    num_epochs = config["optim"]["epochs"]
    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    trn_losses = []
    val_losses = []
    for epoch in progress_bar:
        train_loss = train_regression_nn(train_loader, model, criterion, optimizer, device)
        val_loss = evaluate_model_mse(val_loader, model, device)

        progress_bar.set_postfix(
            {"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"}
        )
        trn_losses.append(train_loss)
        val_losses.append(val_loss)

    test_preds, test_targets = get_mean_preds_and_targets(test_loader, model, device)
    test_mse = get_mse(test_targets, test_preds)
    print("Test MSE: {:.4f}".format(test_mse))

    plt.plot(np.arange(num_epochs), trn_losses, label="TRAIN")
    plt.plot(np.arange(num_epochs), val_losses, label="VAL")
    plt.legend()
    plt.show()

    get_1d_mean_plot(X_test, y_test, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trn-cfg",
        type=str,
        default="./toy_exp_train_config.yaml",
        help="Path to data config file",
    )
    args = parser.parse_args()

    cfg = get_yaml(args.trn_cfg)

    main(config=cfg)
