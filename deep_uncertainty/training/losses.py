import torch


def double_poisson_nll(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the mean double poisson negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a DoublePoissonNN. Dims along last axis are assumed to be (logmu, logphi).
        targets (torch.Tensor): Regression targets for the DoublePoissonNN. Shape: (n, 1).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    eps = 1e-5
    logmu, logphi = torch.split(output, [1, 1], dim=-1)
    phi = torch.exp(logphi)
    mu = torch.exp(logmu)
    losses = (-0.5 * logphi) + phi * mu - (targets * phi * (1 + logmu - torch.log(targets + eps)))
    return losses.mean()


def gaussian_nll(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the mean Gaussian negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a GaussianNN. Dims along last axis are assumed to be (mu, logvar).
        targets (torch.Tensor): Regression targets for the GaussianNN. Shape: (n, 1).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    loss = 0.5 * (torch.exp(-logvar) * (targets - mu) ** 2 + logvar).mean()
    return loss
