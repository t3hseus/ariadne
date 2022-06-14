from collections import Callable

import gin
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np

from ariadne.data_loader import BaseDataLoader
from ariadne.selector.dataset import SelectorDataset


@gin.configurable
class SelectorDataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: SelectorDataset.__class__,
                 valid_size: float,
                 collate_fn=None,
                 max_size=None,
                 with_random=True):
        super(SelectorDataLoader, self).__init__(batch_size)
        self.dataset = dataset(n_samples=max_size)
        data_size = len(self.dataset)
        n_valid = int(data_size * valid_size)
        n_train = data_size - n_valid
        self.collate_fn = collate_fn
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))

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


@gin.configurable('selector_collate_fn')
def collate_fn(samples):
    batch_input_lengths = np.asarray(
        [x[0]['input_lengths'] for x in samples],
        dtype=np.int32
    )
    max_len = batch_input_lengths.max()
    batch_size = len(samples)
    # prepare batch vars
    batch_inputs = np.zeros((batch_size, max_len, samples[0][0]['inputs'].shape[1]), dtype=np.float32)
    batch_targets = np.zeros((batch_size, 2), dtype=np.float32)

    # collect batches
    for i in range(batch_size):
        batch_inputs[i, :batch_input_lengths[i]] = samples[i][0]['inputs']
        batch_targets[i] = samples[i][1]

    return {
        'inputs': torch.from_numpy(batch_inputs),
        'input_lengths': torch.from_numpy(batch_input_lengths)
    }, torch.from_numpy(batch_targets)
