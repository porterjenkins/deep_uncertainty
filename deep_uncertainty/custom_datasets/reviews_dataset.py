from pathlib import Path
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset


class ReviewsDataset(Dataset):
    """Pytorch implementation of the Amazon Reviews dataset (5-core Patio Lawn and Garden subset), originally found at https://nijianmo.github.io/amazon/index.html."""

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
    ):
        """Create an instance of the Reviews dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            split (str): Dataset split to load. Must be "train", "val", or "test".
        """
        self.root_dir = Path(root_dir)
        self.split = split

        if not self._already_downloaded():
            raise Exception(
                "Dataset is not present in the specified location. Contact the authors for access."
            )

        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (self.root_dir / "dataset.json").exists()

    def _get_instances_df(self) -> pd.DataFrame:
        full_instances = pd.read_json(self.root_dir / "dataset.json", orient="records")
        mask = full_instances["split"] == self.split
        return full_instances[mask][["review_text", "rating"]]

    def __getitem__(self, idx: int) -> tuple[str, int]:
        row = self.instances.iloc[idx]
        return row["review_text"], row["rating"]

    def __len__(self):
        return len(self.instances)
