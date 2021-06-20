from collections import Callable

import gin
import torch
from torch.utils.data import random_split, Subset, DataLoader

from ariadne.data_loader import BaseDataLoader
from ariadne.point_net.dataset import PointsDatasetMemory, BatchBucketSampler, SubsetWithItemLen


@gin.configurable
class PointsDataLoader(BaseDataLoader):

    def __init__(self,
                 dataset: PointsDatasetMemory.__class__,
                 collate_fn: Callable,
                 n_train: int,
                 n_valid: int,
                 drop_last: bool = True,
                 with_random=True):
        super(PointsDataLoader, self).__init__(-1)
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

    def get_one_sample(self) -> dict:
        return {'x' :torch.rand(1,3,100) }


@gin.configurable
class PointsDataLoaderNoSampler(BaseDataLoader):

    def __init__(self,
                 dataset: PointsDatasetMemory.__class__,
                 collate_fn: Callable,
                 n_train: int,
                 n_valid: int,
                 batch_size: int,
                 drop_last: bool = True,
                 with_random=True):
        super(PointsDataLoaderNoSampler, self).__init__(-1)
        self.dataset = dataset(n_samples=n_valid + n_train)
        self.batch_size = batch_size
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))
        self.collate_fn = collate_fn

        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn)