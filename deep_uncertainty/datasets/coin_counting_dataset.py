from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class CoinCountingDataset(Dataset):
    """Pytorch implementation of the coin counting dataset, found at https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset?resource=download."""

    def __init__(self, root_dir: str | Path = "./data/coin-counting"):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.instances = self._prepare_instances_df()

    def _prepare_instances_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.root_dir / "coins_count_values.csv")
        df["path"] = str(self.image_dir) + "/" + df["folder"] + "/" + df["image_name"]
        df.drop(["folder", "image_name"], axis="columns", inplace=True)
        return df

    def __getitem__(self, idx: int) -> tuple[PILImage, int]:
        row = self.instances.iloc[idx]
        image = Image.open(row["path"])
        count = row["coins_count"]

        return image, count

    def __len__(self):
        return len(self.instances)
