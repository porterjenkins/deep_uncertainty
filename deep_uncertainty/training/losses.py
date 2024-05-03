import warnings
from math import log
from math import pi

import torch


def double_poisson_nll(
    output: torch.Tensor, targets: torch.Tensor, beta: float | None = None
) -> torch.Tensor:
    """Compute the mean double poisson negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a DoublePoissonNN. Dims along last axis are assumed to be (logmu, logphi).
        targets (torch.Tensor): Regression targets for the DoublePoissonNN. Shape: (n, 1).
        beta (float | None, optional): If specified, the power to raise (1 / `phi`) to for re-scaling the final loss (for gradient disentanglement). Must be between 0 and 1. Defaults to None (no re-scaling).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `double_poisson_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )
    if beta is not None:
        if beta < 0 or beta > 1:
            raise ValueError(f"Invalid value of beta specified. Must be in [0, 1]. Got {beta}")

    logmu, logphi = torch.split(output, [1, 1], dim=-1)

    # Clamp logmu and logphi so the implied mu/phi ratio isn't too small (leading to stability issues).
    ln10 = torch.tensor(10.0, device=logmu.device).log()
    logmu_clamped = torch.clamp(logmu, min=-6.0 * ln10)
    logvar = torch.clamp(logmu_clamped - logphi, min=-4.0 * ln10)
    logphi_clamped = logmu_clamped - logvar

    mu = torch.exp(logmu_clamped)
    phi = torch.exp(logphi_clamped)

    losses = (
        (-0.5 * logphi_clamped)
        + phi * mu
        - phi * (targets + torch.xlogy(targets, mu) - torch.xlogy(targets, targets))
    )

    if beta is not None and beta != 0:
        losses = torch.pow(phi.detach(), -beta) * losses

    return losses.mean()


def gaussian_nll(
    outputs: torch.Tensor, targets: torch.Tensor, beta: float | None = None
) -> torch.Tensor:
    """Compute the mean Gaussian negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a GaussianNN. Dims along last axis are assumed to be (mu, logvar).
        targets (torch.Tensor): Regression targets for the GaussianNN. Shape: (n, 1).
        beta (float | None, optional): If specified, the power to raise (`var`) to for re-scaling the final loss (for gradient disentanglement). Must be between 0 and 1. Defaults to None (no re-scaling).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `gaussian_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )
    if beta is not None:
        if beta < 0 or beta > 1:
            raise ValueError(f"Invalid value of beta specified. Must be in [0, 1]. Got {beta}")

    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    losses = 0.5 * (torch.exp(-logvar) * (targets - mu) ** 2 + logvar)

    if beta is not None and beta != 0:
        var = torch.exp(logvar)
        losses = torch.pow(var.detach(), beta) * losses

    return losses.mean()


def neg_binom_nll(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the mean Negative Binomial negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a NegBinomNN. Dims along last axis are assumed to be (mu, alpha).
        targets (torch.Tensor): Regression targets for the NegBinomNN. Shape: (n, 1).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `double_poisson_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )

    mu, alpha = torch.split(outputs, [1, 1], dim=-1)
    eps = torch.tensor(1e-5, device=alpha.device)
    alpha_recip = torch.pow(torch.maximum(alpha, eps), -1)
    alphamu = torch.maximum(alpha * mu, eps)

    losses = (
        -torch.lgamma(targets + alpha_recip)
        + torch.lgamma(alpha_recip)
        + alpha_recip * torch.log(1 + alphamu)
        - targets * (torch.log(alphamu) - torch.log(1 + alphamu))
    )
    return losses.mean()


def faithful_gaussian_nll(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the modified Gaussian negative log likelihood according to https://arxiv.org/abs/2212.09184 over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a FaithfulGaussianNN. Dims along last axis are assumed to be (mu, logvar).
        targets (torch.Tensor): Regression targets for the FaithfulGaussianNN. Shape: (n, 1).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `faithful_gaussian_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )

    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    dist = torch.distributions.Normal(loc=mu.detach(), scale=logvar.exp().sqrt())

    mse_penalty = 0.5 * (targets - mu) ** 2
    dist_penalty: torch.Tensor = -dist.log_prob(targets)
    losses = mse_penalty + dist_penalty
    return losses.mean()


def natural_gaussian_nll(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the modified natural Gaussian negative log likelihood according to Immer et al '23 (link below):

    https://proceedings.neurips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html

    This implementation is an essential replica of the code provided in the appendix of Immer et al '23.

    Args:
        output (torch.Tensor): The (n, 2) output from a NaturalGaussianNN. Dims along last axis are assumed to be (eta_1, eta_2).
        targets (torch.Tensor): Regression targets for the NaturalGaussianNN. Shape: (n, 1).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `natural_gaussian_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )
    C = -0.5 * log(2 * pi)
    n = len(outputs)
    target = torch.cat([targets, targets.square()], dim=1)
    inner = torch.einsum("nk,nk->n", target, outputs)
    log_A = outputs[:, 0].square() / (4 * outputs[:, 1]) + 0.5 * torch.log(-2 * outputs[:, 1])
    log_lik = n * C + inner.sum() + log_A.sum()
    return -log_lik / n
