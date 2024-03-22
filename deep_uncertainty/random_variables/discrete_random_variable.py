import numpy as np
import torch


class DiscreteRandomVariable:
    """Base class for a discrete random variable, assumed to have support [0, infinity] (truncated at `max_value`).

    Attributes:
        dimension (int): The dimension of the random variable.
        max_value (int, optional): The value at which this random variable's support is truncated. Defaults to 2000.
        use_torch (bool, optional): Whether/not to use a pytorch backend for this class's calculations. Defaults to False (numpy).
    """

    def __init__(
        self,
        dimension: int,
        max_value: int = 2000,
        use_torch: bool = False,
        device: torch.device | None = None,
    ):
        """Initialize a DiscreteRandomVariable.

        Args:
            dimension (int): The dimension of the random variable.
            max_value (int, optional): The value at which this random variable's support is truncated. Defaults to 2000.
            use_torch (bool, optional): Whether/not to use a pytorch backend for this class's calculations. Defaults to False (numpy).
            device (torch.device | None, optional): If using pytorch, which device to perform inference on. Defaults to None.
        """
        self.dimension = dimension
        self.max_value = max_value
        self.use_torch = use_torch
        if use_torch and device is None:
            raise ValueError("Must specify device to perform inference on if using torch.")
        self.device = device
        self._pmf_vals = None
        self._cdf_vals = None

    def pmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        """Compute the probability of this random variable taking the value(s) x.

        Args:
            x (int | np.ndarray | torch.Tensor): The value(s) to compute the probability of.

        Returns:
            float | np.ndarray | torch.Tensor: The probability of x.
        """
        d = self.pmf_vals.shape[1]
        if not self.use_torch:
            result = self.pmf_vals[x, np.arange(d)]
            if result.size == 1:
                return result.item()
        else:
            result = self.pmf_vals[x, torch.arange(d, device=self.device)]
            if result.numel() == 1:
                return result.item()

        return result

    def ppf(self, q: float | torch.Tensor) -> int | float | np.ndarray | torch.Tensor:
        """Return the smallest possible value of this random variable at which the probability mass to the left is greater than or equal to `q`.

        Args:
            q (float | torch.Tensor): The desired quantile.

        Returns:
            int | np.ndarray | torch.Tensor: The smallest value at which this distribution has mass >= `q` to the left of it.
        """
        mask = self.cdf_vals >= min(q, 0.9999)
        if not self.use_torch:
            lowest_val_with_mass_over_q: np.ndarray = np.argmax(mask, axis=0)
            if lowest_val_with_mass_over_q.size == 1:
                return lowest_val_with_mass_over_q.item()
        else:
            lowest_val_with_mass_over_q: torch.Tensor = torch.argmax(mask.long(), dim=0)
            if lowest_val_with_mass_over_q.numel() == 1:
                return lowest_val_with_mass_over_q.item()
        return lowest_val_with_mass_over_q

    def cdf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        """Compute the probability of this random variable taking a value <= x.

        Args:
            x (int | np.ndarray | torch.Tensor): The value whose leftward mass will be returned.

        Returns:
            float | np.ndarray | torch.Tensor: The probability of <= x.
        """
        d = self.cdf_vals.shape[1]
        if not self.use_torch:
            result: np.ndarray = self.cdf_vals[x, np.arange(d)]
            if result.size == 1:
                return result.item()
        else:
            result: torch.Tensor = self.cdf_vals[x, torch.arange(d, device=self.device)]
            if result.numel() == 1:
                return result.item()
        return result

    def rvs(self, size: int | tuple) -> np.ndarray | torch.Tensor:
        """Draw a sample of the given size from this random variable.

        Args:
            size (int | tuple): The size of the sample to return.

        Returns:
            np.ndarray | torch.Tensor: The sample from this random variable.
        """
        if not self.use_torch:
            U = np.random.uniform(size=size)
            draws = np.zeros((U.size, self.dimension))
        else:
            U = torch.rand(size, device=self.device)
            draws = torch.zeros(U.numel(), self.dimension, device=self.device)

        for i, u in enumerate(U.ravel()):
            draws[i] = self.ppf(u)
        draws = draws.reshape(size, -1)
        return draws

    @property
    def pmf_vals(self) -> np.ndarray | torch.Tensor:
        if self._pmf_vals is None:
            truncated_support = (
                np.arange(self.max_value).reshape(-1, 1)
                if not self.use_torch
                else torch.arange(self.max_value, device=self.device).reshape(-1, 1)
            )
            self._pmf_vals = self._pmf(truncated_support)
            self._pmf_vals = self._pmf_vals / self._pmf_vals.sum(axis=0)
        return self._pmf_vals

    @property
    def cdf_vals(self) -> np.ndarray | torch.Tensor:
        if self._cdf_vals is None:
            self._cdf_vals = (
                np.cumsum(self.pmf_vals, axis=0)
                if not self.use_torch
                else torch.cumsum(self.pmf_vals, dim=0)
            )
        return self._cdf_vals

    def expected_value(self) -> float | np.ndarray | torch.Tensor:
        """Return the expected value of this random variable."""
        raise NotImplementedError("Should be implemented by subclass.")

    def _pmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        """Calculate the probability that this random variable takes on the value(s) x. Does not need to be normalized.

        Args:
            x (int | np.ndarray | torch.Tensor): The value(s) to compute the (possibly unnormalized) probability of.

        Returns:
            float | np.ndarray | torch.Tensor: The probability that this distribution takes on the value(s) x.
        """
        raise NotImplementedError("Should be implemented by subclass.")
