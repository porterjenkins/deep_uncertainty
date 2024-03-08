import warnings

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

    eps = torch.tensor(1e-5)
    logmu, logphi = torch.split(output, [1, 1], dim=-1)
    phi = torch.exp(logphi)
    mu = torch.exp(logmu)
    losses = (
        (-0.5 * logphi)
        + phi * mu
        - (targets * phi * (1 + logmu - torch.log(torch.maximum(targets, eps))))
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
    eps = torch.tensor(1e-5)
    alpha_recip = torch.pow(torch.maximum(alpha, eps), -1)
    alphamu = torch.maximum(alpha * mu, eps)

    losses = (
        -torch.lgamma(targets + alpha_recip)
        + torch.lgamma(alpha_recip)
        + alpha_recip * torch.log(1 + alphamu)
        - targets * (torch.log(alphamu) - torch.log(1 + alphamu))
    )
    return losses.mean()
