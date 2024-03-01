import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from deep_uncertainty.evaluation.calibration import compute_mean_calibration
from deep_uncertainty.evaluation.calibration import plot_regression_calibration_curve
from deep_uncertainty.evaluation.old.metrics import get_calibration
from deep_uncertainty.evaluation.old.metrics import get_mse
from deep_uncertainty.evaluation.plotting import plot_posterior_predictive
from deep_uncertainty.models.old.regressors import OldGaussianDNN
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.model_utils import get_gaussian_bounds
from deep_uncertainty.utils.model_utils import get_nll_gaus_loss
from deep_uncertainty.utils.model_utils import inference_with_sigma
from deep_uncertainty.utils.model_utils import train_gaussian_dnn

NUM_MEMBERS = 5


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

    ensemble_preds = np.zeros((len(test_loader), NUM_MEMBERS))
    ensemble_sigmas = np.zeros((len(test_loader), NUM_MEMBERS))

    for i in range(NUM_MEMBERS):
        # Instantiate and train the network
        model_i = OldGaussianDNN().to(device)

        optimizer = optim.Adam(model_i.parameters(), lr=3e-4)

        # Train the network
        num_epochs = config["optim"]["epochs"]
        progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        trn_losses = []
        val_losses = []
        for epoch in progress_bar:
            train_loss = train_gaussian_dnn(train_loader, model_i, optimizer, device)
            val_loss = get_nll_gaus_loss(val_loader, model_i, device)

            progress_bar.set_postfix(
                {"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"}
            )
            trn_losses.append(train_loss)
            val_losses.append(val_loss)

        test_preds, test_sigmas, test_targets, test_inputs = inference_with_sigma(
            test_loader, model_i, device
        )
        ensemble_preds[:, i] = test_preds.data.numpy().flatten()
        ensemble_sigmas[:, i] = np.sqrt(np.exp(test_sigmas.data.numpy().flatten()))

    mu_star = ensemble_preds.mean(1)
    # equation from Lakshminarayanan 2017
    sigma2_star = np.mean(
        np.power(ensemble_sigmas, 2) + np.power(ensemble_preds, 2), axis=1
    ) - np.power(mu_star, 2)
    sigma_star = np.sqrt(sigma2_star)

    # posterior predictive distribution
    ppd = norm(mu_star.flatten(), sigma_star.flatten())

    test_mse = get_mse(test_targets, mu_star)
    print("Test MSE: {:.4f}".format(test_mse))
    upper, lower = get_gaussian_bounds(mu_star, sigma_star, log_var=False)
    test_calib = get_calibration(test_targets.flatten(), upper, lower)
    print("Test Calib: {:.4f}".format(test_calib))
    mean_calib = compute_mean_calibration(test_targets.data.numpy().flatten(), ppd)
    print("Mean Calib: {:.4f}".format(mean_calib))

    plot_posterior_predictive(
        test_inputs.data.numpy().flatten(), test_targets, mu_star, upper, lower
    )
    plot_regression_calibration_curve(test_targets.data.numpy().flatten(), ppd, num_bins=15)


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
