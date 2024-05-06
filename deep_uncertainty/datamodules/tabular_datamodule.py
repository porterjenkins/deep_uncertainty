from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class TabularDataModule(L.LightningDataModule):
    def __init__(
        self, dataset_path: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
    
    # DLC:
    def normalize_with_stats(self, X, mean, std):
        """Normalize a dataset using provided mean and standard deviation."""
        return (X - mean) / std

    def setup(self, stage, normalize=True):
        if normalize:
            data: dict[str, np.ndarray] = np.load(self.dataset_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]
            
            # mean_X_train = X_train.mean(axis=0)  # Column-wise mean
            # std_X_train = X_train.std(axis=0)    # Column-wise standard deviation

            # mean_Y_train = y_train.mean(axis=0)  # Column-wise mean
            # std_Y_train = y_train.std(axis=0)    # Column-wise standard deviation
            
            # # Normalize training data 
            # X_train_norm = (X_train - mean_X_train) / std_X_train

            # # Normalize validation and test data using the same training mean and std
            # X_val_norm = self.normalize_with_stats(X_val, mean_X_train, std_X_train)
            # X_test_norm = self.normalize_with_stats(X_test, mean_X_train, std_X_train)


            # Y_train_norm = self.normalize_with_stats(y_train, mean_Y_train, std_Y_train)
            # Y_test_norm = self.normalize_with_stats(y_val, mean_Y_train, std_Y_train)
            # Y_test_norm = self.normalize_with_stats(y_test, mean_Y_train, std_Y_train)

            # X_train = X_train_norm
            # X_val = X_val_norm
            # X_test = X_test_norm
            
            # y_train = Y_train_norm
            # y_test = Y_test_norm

        
        else:
            data: dict[str, np.ndarray] = np.load(self.dataset_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]


        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            X_val = X_val.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        self.train = TensorDataset(
            torch.Tensor(X_train),
            torch.Tensor(y_train).unsqueeze(1),
        )
        self.val = TensorDataset(
            torch.Tensor(X_val),
            torch.Tensor(y_val).unsqueeze(1),
        )
        self.test = TensorDataset(
            torch.Tensor(X_test),
            torch.Tensor(y_test).unsqueeze(1),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
