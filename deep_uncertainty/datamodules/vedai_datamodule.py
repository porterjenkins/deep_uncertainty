from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from deep_uncertainty.custom_datasets import ImageDatasetWrapper
from deep_uncertainty.custom_datasets import VEDAIDataset


class VEDAIDataModule(L.LightningDataModule):
    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def prepare_data(self) -> None:
        resize = Resize((224, 224))
        augment = AutoAugment()
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        # Split into train/val is somewhat arbitrary since randomization has already occurred.
        orig_train_dataset = VEDAIDataset(self.root_dir, train=True, fold_num=1)
        orig_train_indices = np.arange(len(orig_train_dataset))
        self.train = ImageDatasetWrapper(
            base_dataset=Subset(orig_train_dataset, orig_train_indices[:-100]),
            transforms=train_transforms,
        )
        self.val = ImageDatasetWrapper(
            base_dataset=Subset(orig_train_dataset, orig_train_indices[-100:]),
            transforms=inference_transforms,
        )
        self.test = VEDAIDataset(
            self.root_dir, train=False, fold_num=1, transform=inference_transforms
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
