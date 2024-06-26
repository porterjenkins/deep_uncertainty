import os
from typing import Iterable

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers import DistilBertTokenizer

from deep_uncertainty.custom_datasets import BibleDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


class BibleDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        max_instances: int | None = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.max_instances = max_instances

    def setup(self, stage):
        self.full = BibleDataset(self.root_dir, self.max_instances)

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.full,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    @staticmethod
    def collate_fn(batch: Iterable[tuple[str, int]]) -> tuple[BatchEncoding, torch.LongTensor]:
        all_text = []
        all_targets = []
        for x in batch:
            all_text.append(x[0])
            all_targets.append(x[1])

        input_tokens = tokenizer(
            all_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        targets_tensor = torch.tensor(all_targets)

        return input_tokens, targets_tensor
