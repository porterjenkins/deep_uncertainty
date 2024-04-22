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
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def prepare_data(self):
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
