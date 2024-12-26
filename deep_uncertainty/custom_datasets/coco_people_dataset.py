import os
from pathlib import Path
from typing import Callable
from typing import Literal

import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class COCOPeopleDataset(Dataset):
    """Subset of COCO with images containing people (labeled with the count of people in each image)."""

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ):
        """Create an instance of the COCOPeople dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            split (str): The dataset split to load (train, val, or test).
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and count in __getitem__.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_dir = self.root_dir / "images"
        self.surface_image_path = surface_image_path
        self.annotations_path = self.root_dir / "annotations.csv"

        for dir in self.root_dir, self.image_dir:
            if not dir.exists():
                os.makedirs(dir)

        if not self._already_downloaded():
            raise Exception(
                "Dataset is not present in the specified location. Contact the authors for access."
            )

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (
            self.annotations_path.exists()
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
        )

    def _get_instances_df(self) -> pd.DataFrame:
        annotations = pd.read_csv(
            self.annotations_path, dtype={"image_id": str, "split": str, "count": int}
        )
        mask = annotations["split"] == self.split
        instances = {"image_path": [], "count": []}
        for _, row in annotations[mask].iterrows():
            image_path = str(self.image_dir / f"{row['image_id']}.jpg")
            count = int(row["count"])
            instances["image_path"].append(image_path)
            instances["count"].append(count)
        return pd.DataFrame(instances)

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = row["image_path"]
        image = Image.open(image_path)
        count = row["count"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            count = self.target_transform(count)
        if self.surface_image_path:
            return image, (image_path, count)
        else:
            return image, count

    def __len__(self):
        return len(self.instances)
