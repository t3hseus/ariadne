from typing import Callable

import gin
import torch
from torch.utils.data import DataLoader, Subset, random_split

from ariadne.data_loader import BaseDataLoader
from ariadne.transformer.dataset import CloudDataset, SubsetWithItemLen


@gin.configurable
class CloudDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: CloudDataset.__class__,
        collate_fn: Callable,
        n_train: int,
        n_valid: int,
        with_random: bool = True,
        batch_size: int = 1,
    ):
        super(CloudDataLoader, self).__init__(-1)
        self.dataset = dataset(n_samples=n_valid + n_train)
        self.batch_size = batch_size
        if with_random:
            self.train_data, self.val_data = random_split(
                self.dataset, [n_train, n_valid]
            )
            self.train_data.__class__ = SubsetWithItemLen
            self.val_data.__class__ = SubsetWithItemLen
        else:
            self.train_data = SubsetWithItemLen(self.train_data, range(0, n_train))
            self.val_data = SubsetWithItemLen(
                self.val_data, range(n_train, n_train + n_valid)
            )
        self.collate_fn = collate_fn

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def get_one_sample(self) -> dict:
        return {"x": torch.rand(1, 3, 100)}
