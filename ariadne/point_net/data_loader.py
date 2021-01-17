from collections import Callable

import gin
from torch.utils.data import random_split, Subset, DataLoader

from ariadne.data_loader import BaseDataLoader
from ariadne.point_net.dataset import PointsDatasetMemory, BatchBucketSampler, SubsetWithItemLen


@gin.configurable
class PointsDataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: PointsDatasetMemory.__class__,
                 collate_fn: Callable,
                 n_train: int,
                 n_valid: int,
                 drop_last: bool = True,
                 with_random=True):
        super(PointsDataLoader, self).__init__(batch_size)
        self.dataset = dataset(n_samples=n_valid + n_train)
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
            self.train_data.__class__ = SubsetWithItemLen
            self.val_data.__class__ = SubsetWithItemLen
        else:
            self.train_data = SubsetWithItemLen(self.train_data, range(0, n_train))
            self.val_data = SubsetWithItemLen(self.val_data, range(n_train, n_train + n_valid))

        self.train_sampler = BatchBucketSampler(self.train_data)
        self.val_sampler = BatchBucketSampler(self.val_data)

        self.collate_fn = collate_fn
        self.drop_last = drop_last


    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=1,
            collate_fn=self.collate_fn,
            batch_sampler= self.val_sampler)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=1,
            collate_fn=self.collate_fn,
            batch_sampler= self.train_sampler)
