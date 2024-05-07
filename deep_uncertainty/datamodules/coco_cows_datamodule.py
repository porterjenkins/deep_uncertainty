from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from deep_uncertainty.custom_datasets import COCOCowsDataset


class COCOCowsDataModule(L.LightningDataModule):

    IMG_SIZE = 224

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.surface_image_path = surface_image_path

    def prepare_data(self) -> None:
        # Force images to be downloaded.
        COCOCowsDataset(self.root_dir)

    def setup(self, stage):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, to_tensor, normalize])
        self.full = COCOCowsDataset(
            self.root_dir,
            surface_image_path=self.surface_image_path,
            transform=inference_transforms,
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.full,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
