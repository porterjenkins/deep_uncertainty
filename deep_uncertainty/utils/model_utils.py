from typing import TypeAlias

import numpy as np
import torch
from scipy.optimize import fmin
from scipy.stats import rv_continuous
from scipy.stats import rv_discrete

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


RandomVariable: TypeAlias = rv_discrete | rv_continuous | DiscreteRandomVariable


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

    return mu / (1 - (sig2 / mu))


def get_mean_preds_and_targets_DPR(loader, model, device):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            beta, alpha = model(inputs)
            lambda_i = torch.exp(inputs * beta.squeeze(-1))
            # mean = model(inputs.to(device))
            preds_all.append(lambda_i.cpu())
            targets_all.append(targets)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    return preds_all, targets_all


def get_mean_preds_and_targets(loader, model, device):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            mean = model(inputs.to(device))
            preds_all.append(mean.cpu())
            targets_all.append(targets)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    return preds_all, targets_all


def inference_with_sigma(loader, model, device):
    model.eval()
    preds_all = []
    sigma_all = []
    targets_all = []
    inputs_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            pred = model(inputs.to(device))
            mean, sig = pred
            sigma_all.append(sig.cpu())
            preds_all.append(mean.cpu())
            targets_all.append(targets)
            inputs_all.append(inputs.cpu())

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    sigma_all = torch.cat(sigma_all)
    inputs_all = torch.cat(inputs_all)
    return preds_all, sigma_all, targets_all, inputs_all


def get_gaussian_bounds(
    preds: torch.Tensor, sigmas: torch.Tensor, num_std: float = 1.96, log_var: bool = True
):
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.data.numpy()
    if log_var:
        std_predicted = np.sqrt(np.exp(sigmas))
    else:
        std_predicted = sigmas

    upper = preds + std_predicted * num_std
    lower = preds - std_predicted * num_std

    return upper, lower


def train_regression_nn(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # input: (32, 1), target: (32, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def train_gaussian_dnn(train_loader, model, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        mean, logvar = model(inputs)

        loss = 0.5 * (torch.exp(-logvar) * (targets - mean) ** 2 + logvar).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate_gaussian_dnn(val_loader, model, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mean, logvar = model(inputs)

            loss = 0.5 * (torch.exp(-logvar) * (targets - mean) ** 2 + logvar).mean()
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def get_nll_gaus_loss(val_loader, model, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mean, logvar = model(inputs)

            loss = 0.5 * (torch.exp(-logvar) * (targets - mean) ** 2 + logvar).mean()
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def get_hdi(rv: RandomVariable, p: float = 0.95):
    """Get the p% highest density interval for the given random variable.

    Args:
        rv (RandomVariable): The random variable to get the HDI for.
        p (float): The amount of probability mass desired in the HDI.
    """
    p_inv = 1.0 - p

    def interval_width(low_tail_prob: float):
        return rv.ppf(p + low_tail_prob) - rv.ppf(low_tail_prob)

    hdi_low_tail_prob = fmin(interval_width, p_inv, ftol=1e-8, disp=False)[0]
    return rv.ppf(hdi_low_tail_prob), rv.ppf(p + hdi_low_tail_prob)


def extract_state_dict(model_chkp_path: str, state_dict_path: str):
    """
    Extracts the state dictionary from a saved PyTorch model checkpoint and saves it to a specified path.

    Args:
        model_chkp_path (str): The path to the saved PyTorch model checkpoint.
        state_dict_path (str): The path where the extracted state dictionary should be saved.

    Example:
        extract_state_dict('path/to/model/checkpoint.pth', 'path/to/save/state_dict.pth')

    Note:
        The saved state dictionary can be loaded with `torch.load('path/to/save/state_dict.pth')`.
    """
    m = torch.load(model_chkp_path, map_location="cpu")
    state_dict = m["state_dict"]
    torch.save(state_dict, state_dict_path)
