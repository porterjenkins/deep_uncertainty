import numpy as np
import torch


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

def get_mean_sigma_preds_and_targets(loader, model, device):
    model.eval()
    preds_all = []
    sigma_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            pred = model(inputs.to(device))
            mean, sig = pred
            sigma_all.append(sig.cpu())
            preds_all.append(mean.cpu())
            targets_all.append(targets)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)
    sigma_all = torch.cat(sigma_all)
    return preds_all, sigma_all, targets_all

def get_gaussian_bounds(
        preds: torch.Tensor,
        sigmas: torch.Tensor,
        num_std:float = 1.96,
        log_var:bool = True
):
    if log_var:
        std_predicted = np.sqrt(np.exp(sigmas.squeeze().numpy()))

    upper = preds + std_predicted*np.exp(num_std)
    lower = preds - std_predicted*np.exp(num_std)

    return upper, lower

