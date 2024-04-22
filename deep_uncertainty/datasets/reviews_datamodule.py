import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchtext.datasets import AmazonReviewFull


class ReviewsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        self.train, self.test = AmazonReviewFull(self.data_dir, split=("train", "test"))
        self.train, self.val = random_split(
            dataset=self.train, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(1998)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=9)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, self.batch_size, shuffle=False, num_workers=9)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size, shuffle=False, num_workers=9)
