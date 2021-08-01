import logging
from torch import multiprocessing

from typing import Callable

import gin
from torch.utils.data import random_split, Subset, DataLoader

from ariadne.graph_net.dataset import GraphBatchBucketSampler
from ariadne_v2 import jit_cacher
from ariadne_v2.data_loader import BaseDataLoader
from experiments.graph.dataset import TorchGraphDataset


LOGGER = logging.getLogger('ariadne.dataloader')

@gin.configurable
class GraphDataLoader_NEW(BaseDataLoader):
    def __init__(self,
                 batch_size: int,
                 dataset_cls: TorchGraphDataset.__class__,
                 collate_fn: Callable,
                 subset_cls: Subset.__class__ = Subset,
                 with_random=True):
        # HACK: determine how to transfer mutex with the pytorch multiprocessing between the process
        lock = multiprocessing.Lock()
        jit_cacher.init_locks(lock)
        # END HACK

        super(GraphDataLoader_NEW, self).__init__(batch_size)
        self.subset_cls = subset_cls
        self.dataset = dataset_cls()
        with jit_cacher.instance() as cacher:
            self.dataset.connect(cacher,
                                 self.dataset.dataset_name,
                                 drop_old=False, mode='r')

        assert len(self.dataset) > 0, f"empty dataset {self.dataset.dataset_name}"
        n_train = int(len(self.dataset) * 0.8)
        n_valid = len(self.dataset) - n_train

        LOGGER.info(f"Initializing dataloader with {self.dataset.__class__.__name__}. Len = {len(self.dataset)}.")
        LOGGER.info(f"Num train samples: {n_train}")
        LOGGER.info(f"Num valid samples: {n_valid}")

        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
            self.train_data.__class__ = self.subset_cls
            self.val_data.__class__ = self.subset_cls
        else:
            self.train_data = self.subset_cls(self.dataset, range(0, n_train))
            self.val_data = self.subset_cls(self.dataset, range(n_train, n_train + n_valid))
        self.collate_fn = collate_fn

    def __del__(self):
        if self.dataset:
            self.dataset.disconnect()

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



@gin.configurable
class GraphsDataLoader_Sampler_New(GraphDataLoader_NEW):
    def __init__(self,
                 dataset_cls: TorchGraphDataset.__class__,
                 collate_fn: Callable,
                 subset_cls: Subset.__class__ = Subset,
                 with_random=True):
        super(GraphsDataLoader_Sampler_New, self).__init__(batch_size=1,
                                                           dataset_cls=dataset_cls,
                                                           collate_fn=collate_fn,
                                                           with_random=with_random,
                                                           subset_cls=subset_cls)

        self.train_sampler = GraphBatchBucketSampler(self.train_data)
        self.val_sampler = GraphBatchBucketSampler(self.val_data)

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
