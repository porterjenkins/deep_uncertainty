from typing import List

import torch
from torch import nn


class MLPBackbone(nn.Module):
    """An MLP feature extractor for common use by RegressionNN models.

    Attributes:
        input_dim (int, optional): Dimension of input data. Defaults to 1 (scalars).
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        super(MLPBackbone, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RegressionNN(nn.Module):
    """A neural network for simple regression tasks.

    Attributes:
        input_dim (int, optional): Dimension of input data. Defaults to 1 (scalars).
        output_dim (int, optional): Dimension of outputs. If learning a distribution, these will be the parameters of that distribution. Defaults to 1.
        log_dims (List[bool], optional): If provided, a list specifying, for each output dim, if that output is in log space during training. Defaults to [] (no log outputs).

    Raises:
        ValueError: If `log_dims` does not have the same dimensionality as specified by `output_dim`.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1, log_dims: List[bool] = []):
        super(RegressionNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if log_dims:
            if len(log_dims) != output_dim:
                raise ValueError("Number of elements in `log_dims` should match `output_dim`.")
        self.log_dims = log_dims

        self.backbone = MLPBackbone(input_dim)
        self.head = nn.Linear(self.backbone.output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, `self.input_dim`).

        Returns:
            torch.Tensor: Output tensor, with shape (N, `self.output_dim`).

        If viewing outputs as the parameters of some distribution, use `torch.split(y_hat, [1, 1, ...], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        if self.log_dims and not self.training:
            for j, dim_is_in_log_space in enumerate(self.log_dims):
                if dim_is_in_log_space:
                    y_hat[:, j] = torch.exp(y_hat[:, j])
        return y_hat


"""The code below is maintained solely for legacy scripts and should be deprecated eventually. New scripts should employ the generalized RegressionNN for experiments."""
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class OldRegressionNN(nn.Module):
    def __init__(self, log_output: bool = False):
        super(OldRegressionNN, self).__init__()
        self.log_output = log_output
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        y_hat = self.layers(x)
        if (self.log_output) and (not self.training):
            y_hat = torch.exp(y_hat)
        return y_hat


class OldGaussianDNN(nn.Module):
    def __init__(self):
        super(OldGaussianDNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.mean_output = nn.Linear(64, 1)
        self.logvar_output = nn.Linear(64, 1)

    def forward(self, x):
        h = self.layers(x)
        mean = self.mean_output(h)
        logvar = self.logvar_output(h)
        return mean, logvar


class OldPyroGaussianDNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](1, 32)
        self.fc1.weight = PyroSample(dist.Normal(0.0, 1.0).expand([32, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.0, 1.0).expand([32]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](32, 32)
        self.fc2.weight = PyroSample(dist.Normal(0.0, 1.0).expand([32, 32]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0.0, 1.0).expand([32]).to_event(1))
        self.mean = PyroModule[nn.Linear](32, 1)
        self.mean.weight = PyroSample(dist.Normal(0.0, 1.0).expand([1, 32]).to_event(2))
        self.mean.bias = PyroSample(dist.Normal(0.0, 1.0).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.mean(x).squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 1.0))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
