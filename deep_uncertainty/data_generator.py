from typing import Callable

import numpy as np
import torch
from scipy.stats import nbinom
from scipy.stats import poisson

from deep_uncertainty.random_variables import DiscreteConflation
from deep_uncertainty.random_variables import DoublePoisson


class DataGenerator:
    """An object to generate tabular data.

    All class methods should be implemented as static methods, making this a sort of "factory" class.
    """

    @staticmethod
    def generate_train_val_test_split(
        data_gen_function: Callable[..., tuple[np.ndarray, np.ndarray]],
        data_gen_params: dict,
        split_pcts: list[float] = [0.8, 0.1, 0.1],
        random_seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Pipe a data generation function into a train/val/test split for experiments.

        Args:
            data_gen_function (Callable[..., tuple[np.ndarray, np.ndarray]]): A data generation function that outputs an X and y array.
            data_gen_params (dict): Dictionary with keyword params for `data_gen_function`.
            split_pcts (list[float], optional): Percentage of data to put in each split (in train, val, test order). Defaults to [0.8, 0.1, 0.1].
            random_seed (int | None, optional): Random seed for reproducibility (if desired).

        Raises:
            ValueError: If `split_pcts` does not sum to 1.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys for X_train, X_val, X_test, y_train, y_val, y_test (and the respective arrays).
        """
        if np.sum(split_pcts) != 1:
            raise ValueError("`split_pcts` must sum to 1.")
        X, y = data_gen_function(**data_gen_params)
        n = len(X)
        generator = np.random.default_rng(random_seed)
        indices = np.arange(n)
        indices = generator.permutation(indices)
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
    def generate_count_dataset_with_isolated_points(
        n: int = 498,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create a count dataset similar to the continuous one used in Figure 2 of "Faithful Heteroscedastic Regression with Neural Networks".

        This dataset is created by sampling x ~ Uniform(3, 8). Define mu = ceil(xsin(x)) + 15, phi = (6 - 0.03x^2).
        We sample y ~ DoublePoisson(mu, phi).

        The isolated points (X = 1, Y = ceil(sin(1)) + 15) and (X = 10, Y = ceil(10sin(10)) + 15) are returned separately.

        Returns:
            np.ndarray: X
            np.ndarray: y
        """
        X = np.random.uniform(low=3, high=8, size=n)

        y_mu = np.ceil(X * np.sin(X)) + 15
        y_phi = 6 - 0.03 * X**2
        y = np.array([DoublePoisson(mu, phi).rvs(1) for mu, phi in zip(y_mu, y_phi)])

        isolated_X = np.array([1, 10])
        isolated_y = np.ceil(isolated_X * np.sin(isolated_X)) + 15

        X = np.concatenate([X, isolated_X])
        y = np.concatenate([y, isolated_y])

        return X, y

    @staticmethod
    def generate_discrete_conflation_sine_wave(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        x_vals = []
        y_vals = []
        variances = []
        support = torch.arange(2000)
        lower_bounds = []
        upper_bounds = []
        for _ in range(n):
            x = torch.rand(1) * 2 * np.pi
            rv_list = [torch.distributions.Poisson(rate=10 * np.sin(x) + 10)] * 5
            conflation = DiscreteConflation(rv_list)

            probs = conflation.pmf(support)

            mu = (probs * support).sum()
            var = (probs * (support.unsqueeze(1) - mu).transpose(0, 1) ** 2).sum(dim=1)
            variances.append(var)
            lower_bounds.append(-conflation.ppf(0.025) + 30)
            upper_bounds.append(-conflation.ppf(0.975) + 30)

            y = conflation.rvs((1, 1))

            x_vals.append(x)
            y_vals.append(-y + 30)

        X = np.array(x_vals)
        y = np.array(y_vals)
        np.savez(
            "data/discrete-wave/gt_uncertainty.npz",
            X=X,
            y=y,
            lower=np.array(lower_bounds),
            upper=np.array(upper_bounds),
            aleatoric=np.array(variances),
        )

        return X, y

    @staticmethod
    def generate_nbinom_bowtie(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(-3, 3, size=n)
        dist = nbinom(n=X**2, p=0.5)
        y = dist.rvs()
        return X, y

    @staticmethod
    def generate_poisson_parabola(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(-5, 5, size=n)
        dist = poisson(np.exp(0.5 * X))
        y = dist.rvs()
        return X, y
