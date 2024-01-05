from typing import Tuple

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample
from torch import nn


class RegressionNN(nn.Module):
    def __init__(self, log_output: bool = False):
        super(RegressionNN, self).__init__()
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

class GaussianDNN(nn.Module):
    def __init__(self):
        super(GaussianDNN, self).__init__()

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

class PyroGaussianDNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](1, 32)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([32, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](32, 32)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([32, 32]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))
        self.mean = PyroModule[nn.Linear](32, 1)
        self.mean.weight = PyroSample(dist.Normal(0., 1.).expand([1, 32]).to_event(2))
        self.mean.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.mean(x).squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
    
class DoublePoissonNN(nn.Module):
    """A neural network that learns the parameters of a double poisson to fit its regression targets."""
    
    def __init__(self):
        super(DoublePoissonNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor, with shape (n, d).

        Returns:
            torch.Tensor: Tensor with shape (n, 2), where the outputs along dimension 2 are logmu, logphi. Run `torch.split(output, [1, 1], dim=-1)` to separate into `logmu`, `logphi` tensors.
        """
        return self.layers(x)
