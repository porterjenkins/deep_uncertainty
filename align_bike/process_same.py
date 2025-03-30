import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import sys
# from __future__ import annotations

sys.path.append("/home/local/ASURITE/longchao/Desktop/project/CountUncertainty/officialImplement/")  # or adjust to your actual project root


from typing import List, Tuple, TypeVar
import torch
from torch.utils.data import Dataset, Subset

T = TypeVar("T")

import torch


class StandardScaler:
    """
    Standard scaler for tabular data.
    """
    mean_: torch.Tensor
    std_: torch.Tensor

    def fit(self, X: torch.Tensor) -> "StandardScaler":

        """
        Fits the mean and std of this scaler via the provided tabular data.
        """
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided tabular data with the mean and std that was fitted previously.
        """
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of tabular data.
        """
        return X * self.std_ + self.mean_





def tabular_train_test_split(
    *tensors: torch.Tensor,
    train_size: float,
    generator: torch.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Splits the given tensors randomly into training and test tensors. Each tensor is split with
    the same indices.

    Args:
        tensors: The tensors to split. Must all have the same number of elements in the first
            dimension.
        train_size: The fraction in ``(0, 1)`` to use for the training data.
        generator: The generator to use for generating train/test splits.

    Returns:
        The tensors split into training and test tensors.
    """
    num_items = tensors[0].size(0)
    num_train = round(num_items * train_size)
    permutation = torch.randperm(num_items, generator=generator)
    return [(t[permutation[:num_train]], t[permutation[num_train:]]) for t in tensors]




def main(csv_path: str):
    df = pd.read_csv(csv_path)

    # Group data by season
    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for season_idx, df_season in df.groupby("season"):
        x = torch.from_numpy(df_season.iloc[:, 3:-3].to_numpy()).float()
        y = torch.from_numpy(df_season["cnt"].to_numpy()).float()
        data[season_map[int(season_idx)]] = (x, y)

    # Split summer into train / val / test
    (X_train, X_test), (y_train, y_test) = tabular_train_test_split(
        *data["summer"], train_size=0.8, generator=torch.Generator().manual_seed(42)
    )
    (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
        X_train, y_train, train_size=0.8, generator=torch.Generator().manual_seed(42)
    )

    # Normalize inputs and outputs
    input_scaler = StandardScaler().fit(X_train)
    output_scaler = StandardScaler().fit(y_train)

    X_train = input_scaler.transform(X_train)
    X_val = input_scaler.transform(X_val)
    X_test = input_scaler.transform(X_test)

    y_train = output_scaler.transform(y_train)
    y_val = output_scaler.transform(y_val)
    y_test = output_scaler.transform(y_test)

    print("✅ Data successfully split and normalized!\n")
    print("X_train:", X_train.shape, "\n", X_train[:3])
    print("y_train:", y_train.shape, "\n", y_train[:3])
    print("X_val:", X_val.shape, "\n", X_val[:3])
    print("y_val:", y_val.shape, "\n", y_val[:3])
    print("X_test:", X_test.shape, "\n", X_test[:3])
    print("y_test:", y_test.shape, "\n", y_test[:3])


if __name__ == "__main__":
    # Change this to use day.csv if needed
    csv_path = "/home/local/ASURITE/longchao/Desktop/project/CountUncertainty/officialImplement/base/deep_uncertainty/data/hour.csv"
    main(csv_path)
