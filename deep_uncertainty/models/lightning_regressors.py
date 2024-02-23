import lightning as L
import torch
from torch import nn
from torch import optim

from deep_uncertainty.models.backbones import MLPBackbone


def double_poisson_nll(
    output: torch.Tensor, targets: torch.Tensor, beta: float = 0.0
) -> torch.Tensor:
    """Compute the mean double poisson negative log likelihood over the given targets.

    Args:
        output (torch.Tensor): The (n, 2) output from a DoublePoissonNN. Dims along last axis are assumed to be logmu, logphi.
        targets (torch.Tensor): Regression targets for the DoublePoissonNN. Shape: (n, 1).
        beta (float): Value between 0 and 1 specifying the exponent on the stabilizing (1 / phi) term. Defaults to 0 (standard NLL).

    Returns:
        torch.Tensor: Avg. loss across all targets. Zero-dimensional tensor (torch.Size([])).
    """
    eps = 1e-5
    logmu, logphi = torch.split(output, [1, 1], dim=-1)
    phi = torch.exp(logphi)
    mu = torch.exp(logmu)
    stabilizing_term = phi.detach() ** (-beta)
    losses = stabilizing_term * (
        (-0.5 * logphi) + phi * mu - (targets * phi * (1 + logmu - torch.log(targets + eps)))
    )
    return losses.mean()


class DoublePoissonNN(L.LightningModule):
    """A neural network that learns the parameters of a Double Poisson distribution over each regression target.

    Attributes:
        input_dim (int, optional): Dimension of input data. Defaults to 1 (scalars).
        beta (float, optional)
        optimizer (optim.Optimizer, optional)
        optim_kwargs (dict, optional)
    """

    def __init__(
        self,
        input_dim: int = 1,
        beta: float = 0.0,
        optimizer: optim.Optimizer = optim.AdamW,
        optim_kwargs: dict = {"lr": 1e-3, "weight_decay": 1e-5},
    ):
        super(DoublePoissonNN, self).__init__()

        self.input_dim = input_dim
        self.beta = beta
        self.optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self.backbone = MLPBackbone(input_dim)
        self.head = nn.Linear(self.backbone.output_dim, 2)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self._forward_impl(x)
        loss = double_poisson_nll(y_hat, y, self.beta)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self._forward_impl(x)
        return torch.exp(y_hat)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optim_kwargs)
        return optimizer

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (logmu, logphi), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat
