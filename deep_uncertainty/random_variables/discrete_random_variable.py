from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


class DiscreteRandomVariable:
    """Base class for a discrete random variable, assumed to have support [0, infinity] (truncated at `max_value`).

    To subclass a `DiscreteRandomVariable`, simply implement the internal `_pmf` and `_expected_value` methods.
    Other methods may be overridden when necessary (such as when analytical solutions exist for the cdf).

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
            x (int | np.ndarray | torch.Tensor): The value(s) to compute the probability of. Shape: (m,).

        Returns:
            float | np.ndarray | torch.Tensor: The probability of x. Shape: (m, `self.dimension`).
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

    def ppf(self, q: float | np.ndarray | torch.Tensor) -> int | float | np.ndarray | torch.Tensor:
        """Return the smallest possible value of this random variable at which the probability mass to the left is greater than or equal to `q`.

        If q is a scalar, returns a (`self.dimension`,) array/tensor with the inverse cdf values at q for each distribution represented by this random variable.
        If q is a (`self.dimension`,) array/tensor, returns a (`self.dimension`,) array/tensor with inverse cdf values corresponding positionwise to the q entries.

        Args:
            q (float | np.ndarray | torch.Tensor): The desired quantile(s). Must either be a scalar or a (`self.dimension`,) array/tensor.

        Raises:
            ValueError: If q is the wrong shape.

        Returns:
            int | np.ndarray | torch.Tensor: The smallest value(s) at which this distribution has mass >= `q` to the left of it.
        """
        if not isinstance(q, Iterable):
            q = np.array([q]) if not self.use_torch else torch.tensor(q, device=self.device)

        if q.shape not in {(1,), (self.dimension,)}:
            raise ValueError(
                f"Invalid shape for q. Expected (1,) or ({self.dimension},) but got {q.shape}."
            )

        # Make sure q is slightly less than 1 (rounding issues can lead to bad results when q is exactly 1).
        q = (
            np.minimum(q, 0.9999)
            if not self.use_torch
            else torch.minimum(q, torch.tensor(0.9999, device=self.device))
        )

        mask = self.cdf_vals >= q
        if not self.use_torch:
            inverse_cdf = np.array(np.argmax(mask, axis=0)).squeeze()
            is_scalar = inverse_cdf.size == 1
        else:
            inverse_cdf = torch.argmax(mask.long(), dim=0)
            is_scalar = inverse_cdf.numel() == 1

        if is_scalar:
            return inverse_cdf.item()
        return inverse_cdf

    def cdf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        """Compute the probability of this random variable taking a value <= x.

        Args:
            x (int | np.ndarray | torch.Tensor): The value(s) whose leftward mass will be returned. Shape: (m,).

        Returns:
            float | np.ndarray | torch.Tensor: The probability of <= x. Shape: (m, `self.dimension`).
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

    def rvs(self, size: int | tuple, verbose: bool = False) -> int | np.ndarray | torch.Tensor:
        """Draw a sample of the given size from this random variable.

        If `self.dimension` > 1, the `size` argument must be (n, `self.dimension`) (corresponding to n samples along each dimension).

        Args:
            size (int | tuple): The size of the sample to return.
            verbose (bool, optional): Whether/not to show a progress bar for the sampling procedure.

        Raises:
            ValueError: If the requested `size` is incorrectly specified.

        Returns:
            int | np.ndarray | torch.Tensor: The sample from this random variable.
        """
        if self.dimension > 1:
            if not isinstance(size, tuple) or size[1] != self.dimension:
                raise ValueError(
                    f"Must specify a size compatible with this RV's dimension. Received {size} but expected tuple with shape (n, {self.dimension})."
                )

        size = (size,) if not isinstance(size, tuple) else size
        if not self.use_torch:
            U = np.random.uniform(size=size)
            draws = np.zeros(shape=size)
        else:
            U = torch.rand(*size, device=self.device)
            draws = torch.zeros(size=size, device=self.device)

        n = size[0]
        iterable = range(n) if not verbose else tqdm(range(n), desc="Sampling...", total=n)
        for i in iterable:
            draws[i] = self.ppf(U[i])

        if self.use_torch:
            draws = draws.long()
            is_scalar_draw = draws.numel() == 1
        else:
            draws = draws.astype(int)
            is_scalar_draw = draws.size == 1
        return draws.item() if is_scalar_draw else draws

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

    @property
    def expected_value(self) -> float | np.ndarray | torch.Tensor:
        return self._expected_value()

    def _expected_value(self) -> float | np.ndarray | torch.Tensor:
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
