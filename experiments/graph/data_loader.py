import logging
from torch import multiprocessing

from typing import Callable

import gin
from torch.utils.data import random_split, Subset, DataLoader

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
                 with_random=True):
        # HACK: determine how to transfer mutex with the pytorch multiprocessing between the process
        lock = multiprocessing.Lock()
        jit_cacher.init_locks(lock)
        # END HACK

        super(GraphDataLoader_NEW, self).__init__(batch_size)

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
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))
        self.collate_fn = collate_fn

    def __del__(self):
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