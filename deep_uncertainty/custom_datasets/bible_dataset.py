import os
from pathlib import Path

import pandas as pd
import wget
from torch.utils.data import Dataset


class BibleDataset(Dataset):
    """Pytorch dataset whose items are the verses of the English translation of the King James Version of the Bible.

    Attributes:
        root_dir (Path): Root directory where dataset files are stored.
    """

    DOWNLOAD_URL = "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-eng-kjv.txt"

    def __init__(
        self,
        root_dir: str | Path,
        max_verses: int | None = None,
    ):
        """Create an instance of the BibleDataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            max_verses (int | None, optional): Max number of verses to contain in dataset. Defaults to None.
        """
        self.root_dir = Path(root_dir)
        self.max_verses = max_verses
        if not self.root_dir.exists():
            os.makedirs(self.root_dir)

        if not self._already_downloaded():
            self._download()

        self.verses = self._get_verses()

    def _already_downloaded(self) -> bool:
        return (self.root_dir / "eng-eng-kjv.txt").exists()

    def _download(self):
        wget.download(self.DOWNLOAD_URL, str(self.root_dir))

    def _get_verses(self) -> pd.DataFrame:
        verses = []
        with open(self.root_dir / "eng-eng-kjv.txt", "r") as f:
            num_instances = 0
            for verse in f:
                if verse.strip():
                    verses.append(verse.strip())
                    num_instances += 1
                    if num_instances > (self.max_verses or float("inf")):
                        break
        return verses

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.verses[idx], -1

    def __len__(self):
        return len(self.verses)
