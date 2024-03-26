import os
from pathlib import Path
from shutil import unpack_archive
from typing import Callable

import pandas as pd
import wget
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class VEDAIDataset(Dataset):
    """Pytorch implementation of the VEDAI dataset (1024x1024 color version), found at https://downloads.greyc.fr/vedai/.

    Although labels are originally in context of object detection, in this dataset, they are converted to counts (the number of vehicles in an image).

    Attributes:
        root_dir (Path): Root directory where dataset files are stored.
        annotations_dir (Path): Directory where original dataset annotations are stored.
        images_dir (Path): Directory where original images are stored.
        instances (pd.DataFrame): DataFrame listing image paths and their respective vehicle count.
        train (bool): Whether this dataset represents the train (or test) split of the data.
        fold_num (int): Which fold this dataset represents (in [1, 10]).
        transform (Callable): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`.
        target_transform (Callable): A function/transform that takes in the target and transforms it.
    """

    ANNOTATIONS_URL = "https://downloads.greyc.fr/vedai/Annotations1024.tar"
    IMAGE_URLS = [f"https://downloads.greyc.fr/vedai/Vehicules1024.tar.00{i}" for i in range(1, 6)]

    def __init__(
        self,
        root_dir: str | Path,
        train: bool = True,
        fold_num: int = 1,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
    ):
        """Create an instance of the VEDAI dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            train (bool, optional): Flag to use train (or test) split of data. Defaults to True.
            fold_num (int, optional): Which dataset fold to use (must be in [1, 10]). Defaults to 1.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        """
        if fold_num < 1 or fold_num > 10:
            raise ValueError("Invalid fold number specified. Must give a number in [1, 10].")

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            os.makedirs(self.root_dir)
        if not self._already_downloaded():
            self._download()

        self.annotations_dir = self.root_dir / "Annotations1024"
        self.image_dir = self.root_dir / "Vehicules1024"
        self.train = train
        self.fold_num = fold_num
        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (self.root_dir / "Annotations1024").exists() and (
            self.root_dir / "Vehicules1024"
        ).exists()

    def _download(self):
        annotations_tar_path = wget.download(self.ANNOTATIONS_URL, str(self.root_dir))
        [wget.download(url, str(self.root_dir)) for url in self.IMAGE_URLS]

        combined_image_tar_path = self.root_dir.absolute() / "Vehicules1024.tar"
        os.system(
            f"cat {self.root_dir.absolute() / 'Vehicules1024.tar.*'} > {combined_image_tar_path}"
        )

        for tar_path in [annotations_tar_path, combined_image_tar_path]:
            unpack_archive(filename=tar_path, extract_dir=self.root_dir, format="tar")

        os.system(f"rm {self.root_dir.absolute() / '*tar*'}")
        os.system(f"rm {self.root_dir.absolute() / 'Vehicules1024' / '*_ir.png'}")

    def _get_instances_df(self) -> pd.DataFrame:
        fold_file = (
            self.annotations_dir / f"fold{self.fold_num:02d}{'test' if not self.train else ''}.txt"
        )
        with open(fold_file, "r") as f:
            images_in_fold = [x.strip() for x in f.readlines()]
        instances = []
        for img in images_in_fold:
            image_path = self.image_dir / f"{img}_co.png"
            with open(self.annotations_dir / f"{img}.txt", "r") as f:
                count = len(f.readlines())
            instances.append({"image_path": image_path, "count": count})
        return pd.DataFrame(instances)

    def __getitem__(self, idx: int) -> tuple[PILImage, int]:
        row = self.instances.iloc[idx]
        image = Image.open(row["image_path"])
        count = row["count"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            count = self.target_transform(count)
        return image, count

    def __len__(self):
        return len(self.instances)
