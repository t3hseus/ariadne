from collections import Callable

import gin
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from ariadne.data_loader import BaseDataLoader
from ariadne.tracknet_v2.dataset import TrackNetV2Dataset


@gin.configurable
class TrackNetV2DataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: TrackNetV2Dataset.__class__,
                 n_train: int,
                 n_valid: int,
                 with_random=True):
        super(TrackNetV2DataLoader, self).__init__(batch_size)
        self.dataset = dataset(n_samples=n_valid + n_train)
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size)
