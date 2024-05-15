from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.mnist_test = MNIST(self.root_dir, train=False, download=True, transform=transform)
        self.mnist_predict = MNIST(self.root_dir, train=False, download=True, transform=transform)
        mnist_full = MNIST(self.root_dir, train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(1998)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
