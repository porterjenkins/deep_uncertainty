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
    if beta is not None:
        if beta < 0 or beta > 1:
            raise ValueError(f"Invalid value of beta specified. Must be in [0, 1]. Got {beta}")

    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    losses = 0.5 * (torch.exp(-logvar) * (targets - mu) ** 2 + logvar)

    if beta is not None and beta != 0:
        var = torch.exp(logvar)
        losses = torch.pow(var.detach(), beta) * losses

    return losses.mean()
