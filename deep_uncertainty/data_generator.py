from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import binom

from deep_uncertainty.utils.model_utils import get_binom_n
from deep_uncertainty.utils.model_utils import get_binom_p


class DataGenerator:
    """An object to generate data.

    All class methods should be implemented as static methods, making this a sort of "factory" class.
    """

    @staticmethod
    def generate_train_test_val_split(
        data_gen_function: Callable[..., Tuple[np.ndarray, np.ndarray]],
        data_gen_params: dict,
        split_pcts: List[float] = [0.8, 0.1, 0.1],
    ) -> Dict[str, np.ndarray]:
        """Pipe a data generation function into a train/val/test split for experiments.

        Args:
            data_gen_function (Callable[..., Tuple[np.ndarray, np.ndarray]]): A data generation function that outputs an X and y array.
            data_gen_params (dict): Dictionary with keyword params for `data_gen_function`.
            split_pcts (List[float], optional): Percentage of data to put in each split (in train, val, test order). Defaults to [0.8, 0.1, 0.1].

        Raises:
            ValueError: If `split_pcts` does not sum to 1.

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys for X_train, X_val, X_test, y_train, y_val, y_test (and the respective arrays).
        """
        if np.sum(split_pcts) != 1:
            raise ValueError("`split_pcts` must sum to 1.")
        X, y = data_gen_function(**data_gen_params)
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        train_cutoff = int(split_pcts[0] * n)
        val_cutoff = int((split_pcts[0] + split_pcts[1]) * n)
        train_indices, val_indices, test_indices = np.split(indices, [train_cutoff, val_cutoff])
        return {
            "X_train": X[train_indices],
            "X_val": X[val_indices],
            "X_test": X[test_indices],
            "y_train": y[train_indices],
            "y_val": y[val_indices],
            "y_test": y[test_indices],
        }

    @staticmethod
    def generate_gaussian_sine_wave(
        n: int, x_min: float, x_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a dataset of (x, y) values where y = sin(x) + epsilon, epsilon ~ N(x, 0.5*(1 + e^-x)^-1).

        The variance in y increases with x.

        Args:
            n (int): The number of data points to draw between x_min and x_max.
            x_min (float): Minimum x value to draw data from.
            x_max (float): Maximum x value to draw data from.

        Returns:
            np.ndarray: The x values of the resultant dataset.
            np.ndarray: The y values of the resultant dataset.
        """
        x_values = np.random.uniform(x_min, x_max, n)
        mu_values = np.sin(x_values)
        sigma_values = 0.5 * (1 + np.exp(-x_values)) ** -1
        y_values = np.random.normal(mu_values, sigma_values**2)
        return x_values, y_values

    @staticmethod
    def generate_linear_binom_data(
        n: int,
        x_min: float,
        x_max: float,
        beta_mu: float = 1.5,
        beta_sig: float = 1.1,
        bias_sig: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_values = np.random.uniform(x_min, x_max, n)
        mu = beta_mu * x_values
        sig2 = beta_sig * x_values + bias_sig
        n = np.round(get_binom_n(mu=mu, sig2=sig2)).astype(int)
        p = get_binom_p(mu=mu, n=n)
        y = binom.rvs(n=n, p=p)
        return x_values, y

    @staticmethod
    def generate_nonlinear_binom_data(
        n: int, x_min: float, x_max: float, beta_sig: float = 1.1, bias_sig: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_values = np.random.uniform(x_min, x_max, n)
        mu = np.sin(x_values)
        sig2 = beta_sig * x_values + bias_sig
        n = np.round(get_binom_n(mu=mu, sig2=sig2)).astype(int)
        p = get_binom_p(mu=mu, n=n)
        y = binom.rvs(n=n, p=p)
        return x_values, y

    @staticmethod
    def generate_nonlinear_count_data(
        n: int, x_min: float, x_max: float, n_grid: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_values = np.array(sorted(np.random.uniform(x_min, x_max, n)))
        mu_values = np.sin(x_values + 5) * 10 + 2 + 1.5 * x_values

        mu_min = mu_values.min()
        mu_max = mu_values.max()
        # bin width
        w = (mu_max - mu_min) / n_grid
        idx = (np.array(mu_values) - mu_min) // w
        noise = np.arange(1, n_grid + 2)
        band = np.take(noise, idx.astype(int))
        low = mu_values - 0.5 * band
        high = mu_values + 0.5 * band
        y = np.random.randint(low=low, high=high + 1)

        return x_values, y

    @staticmethod
    def generate_linear_count_data(
        n: int, x_min: float, x_max: float, n_grid: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_values = np.array(sorted(np.random.uniform(x_min, x_max, n)))
        mu_values = 1.5 * x_values

        mu_min = mu_values.min()
        mu_max = mu_values.max()
        # bin width
        w = (mu_max - mu_min) / n_grid
        idx = (np.array(mu_values) - mu_min) // w
        noise = np.arange(1, n_grid + 2)
        band = np.take(noise, idx.astype(int))
        low = mu_values - 0.5 * band
        high = mu_values + 0.5 * band
        y = np.random.randint(low=low, high=high + 1)

        return x_values, y