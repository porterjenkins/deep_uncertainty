import gzip
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import wget
from torch.utils.data import Dataset


class ReviewsDataset(Dataset):
    """Pytorch implementation of the Amazon Reviews dataset (5-core Patio Lawn and Garden subset), found at https://nijianmo.github.io/amazon/index.html.


    Attributes:
        root_dir (Path): Root directory where dataset files are stored.
    """

    DOWNLOAD_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Patio_Lawn_and_Garden_5.json.gz"

    def __init__(
        self,
        root_dir: str | Path,
    ):
        """Create an instance of the VEDAI dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.

        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            os.makedirs(self.root_dir)

        if not self._already_downloaded():
            self._download()

        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (self.root_dir / "Patio_Lawn_and_Garden_5.json").exists()

    def _download(self):
        gz_filename = wget.download(self.DOWNLOAD_URL, str(self.root_dir))
        target_filename = "Patio_Lawn_and_Garden_5.json"
        with gzip.open(self.root_dir / gz_filename, "rb") as f_in:
            with open(self.root_dir / target_filename, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.system(f"rm {self.root_dir.absolute() / gz_filename}")

    def _get_instances_df(self) -> pd.DataFrame:
        instances = {"review_text": [], "rating": []}
        html_pattern = re.compile(r"<[^>]+>")
        with open(self.root_dir / "Patio_Lawn_and_Garden_5.json", "r") as f:
            for line in f:
                review: dict = json.loads(line)
                review_text, rating = review.get("reviewText"), review.get("overall")

                if (
                    review_text is None
                    or rating is None
                    or html_pattern.search(review_text) is not None
                ):
                    continue
                instances["review_text"].append(review_text)
                instances["rating"].append(rating)

        return pd.DataFrame(instances)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        row = self.instances.iloc[idx]
        return row["review_text"], row["rating"]

    def __len__(self):
        return len(self.instances)
