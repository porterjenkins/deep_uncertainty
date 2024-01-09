import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from deep_uncertainty.evaluation.calibration import compute_average_calibration_score
from deep_uncertainty.evaluation.calibration import plot_regression_calibration_curve
from deep_uncertainty.evaluation.metrics import get_calibration
from deep_uncertainty.evaluation.metrics import get_mse
from deep_uncertainty.evaluation.plots import get_sigma_plot_from_test
from deep_uncertainty.models.regressors import OldGaussianDNN
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.model_utils import get_gaussian_bounds
from deep_uncertainty.utils.model_utils import get_nll_gaus_loss
from deep_uncertainty.utils.model_utils import inference_with_sigma
from deep_uncertainty.utils.model_utils import train_gaussian_dnn

NUM_MEMBERS = 5


def get_conflation_pair(mu1: float, mu2: float, sig1: float, sig2: float):
    mu = (sig2 * mu1 + sig1 * mu2) / (sig1 + sig2)
    var = sig1 * sig2 / (sig1 + sig2)
    return mu, var


def get_conflation_2darray(mus: np.ndarray, sigmas: np.ndarray):
    assert mus.shape[1] == sigmas.shape[1]
    m = mus.shape[1]

    for i in range(m - 1):
        m1 = mus[:, i]
        m2 = mus[:, i + 1]
        s1 = sigmas[:, i]
        s2 = sigmas[:, i + 1]
        if i == 0:
            mu_star = m1
            var_star = s1
        mu_star, var_star = get_conflation_pair(mu_star, m2, var_star, s2)

    return mu_star, var_star


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

    ensemble_vars = np.power(ensemble_sigmas, 2)
    mu_star, var_star = get_conflation_2darray(mus=ensemble_preds, sigmas=ensemble_vars)
    sigma_star = np.sqrt(var_star)

    # posterior predictive distribution
    ppd = norm(mu_star.flatten(), sigma_star.flatten())

    test_mse = get_mse(test_targets, mu_star)
    print("Test MSE: {:.4f}".format(test_mse))
    upper, lower = get_gaussian_bounds(mu_star, sigma_star, log_var=False)
    test_calib = get_calibration(test_targets.flatten(), upper, lower)
    print("Test Calib: {:.4f}".format(test_calib))
    mean_calib = compute_average_calibration_score(test_targets.data.numpy().flatten(), ppd)
    print("Mean Calib: {:.4f}".format(mean_calib))

    """for i in range(NUM_MEMBERS):
        upper_i, lower_i = get_gaussian_bounds(ensemble_preds[:, i], ensemble_sigmas[:, i], log_var=False)
        get_sigma_plot_from_test(
            test_inputs.data.numpy().flatten(),
            y_test,
            ensemble_preds[:, i],
            upper_i,
            lower_i,
            title=f'member: {i}'
        )"""

    get_sigma_plot_from_test(
        test_inputs.data.numpy().flatten(), test_targets, mu_star, upper, lower, title="Conflation"
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
