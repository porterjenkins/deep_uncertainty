from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np


class DataGenerator:
    """An object to generate data.

    All class methods should be implemented as static methods, making this a sort of "factory" class.
    """

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
