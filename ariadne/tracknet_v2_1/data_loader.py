from collections import Callable

import gin
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from ariadne.data_loader import BaseDataLoader

@gin.configurable
class EventWiseSplitDataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 train_dataset,
                 valid_dataset,
                 max_size=None):
        super(EventWiseSplitDataLoader, self).__init__(batch_size)
        self.train_data = train_dataset(n_samples=max_size)
        self.val_data = valid_dataset(n_samples=max_size)

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size)
