import numpy as np
import torch
from scipy.special import loggamma
from scipy.special import xlogy

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DoublePoisson(DiscreteRandomVariable):
    """A Double Poisson random variable.

    Attributes:
        mu (float | int | np.ndarray | torch.Tensor): The mu parameter of the distribution (can be vectorized).
        phi (float | int | np.ndarray | torch.Tensor): The phi parameter of the distribution (can be vectorized).
        max_value (int, optional): The highest value to assume support for (since the DPO support is infinite). For numerical purposes. Defaults to 2000.
        dimension (int): The dimension of this random variable (matches the dimension of mu and phi).
        use_torch (bool): Whether/not to use a torch backend (defaults to numpy). Determined by the type of `mu`.
    """

    def __init__(
        self,
        mu: float | int | np.ndarray | torch.Tensor,
        phi: float | int | np.ndarray | torch.Tensor,
        max_value: int = 2000,
    ):
        """Initialize a Double Poisson random variable.

        Args:
            mu (float | int | np.ndarray | torch.Tensor): The mu parameter of the distribution (can be vectorized).
            phi (float | int | np.ndarray | torch.Tensor): The phi parameter of the distribution (can be vectorized).
            max_value (int, optional): The highest value to assume support for (since the DPO support is infinite). For numerical purposes. Defaults to 2000.
        """
        if isinstance(mu, torch.Tensor):
            use_torch = True
            device = mu.device
        else:
            use_torch = False
            device = None

        if not use_torch:
            self.mu = np.clip(np.array(mu), a_min=1e-6, a_max=None)
            self.phi = np.array(phi)
            var = np.clip(np.array(mu) / np.array(phi), a_min=1e-4, a_max=None)
            self.phi = self.mu / var
            dimension = self.mu.size
        else:
            self.mu = torch.clamp(mu, min=1e-6)
            var = torch.clamp(mu / phi, min=1e-4)
            self.phi = self.mu / var
            dimension = self.mu.numel()

        super().__init__(
            dimension=dimension, max_value=max_value, use_torch=use_torch, device=device
        )

    def _expected_value(self) -> float | np.ndarray | torch.Tensor:
        """Return the expected value of this DoublePoisson distribution."""
        return self.mu

    def _pmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        exp = np.exp if not self.use_torch else torch.exp
        return exp(self._logpmf(x))

    def _logpmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:

        if not self.use_torch:
            x = np.array(x)
            return (
                0.5 * np.log(self.phi)
                - self.phi * self.mu
                - x
                + xlogy(x, x)
                - loggamma(x + 1)
                + self.phi * (x + xlogy(x, self.mu) - xlogy(x, x))
            )
        else:
            return (
                0.5 * self.phi.log()
                - self.phi * self.mu
                - x
                + torch.xlogy(x, x)
                - torch.lgamma(x + 1)
                + self.phi * (x + torch.xlogy(x, self.mu) - torch.xlogy(x, x))
            )
