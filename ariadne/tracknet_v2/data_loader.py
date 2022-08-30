from collections import Callable

import gin
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np

from ariadne.data_loader import BaseDataLoader
from ariadne.tracknet_v2.dataset import TrackNetV2Dataset


@gin.configurable
class TrackNetV2DataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: TrackNetV2Dataset.__class__,
                 valid_size: float,
                 collate_fn=None,
                 max_size=None,
                 with_random=True):
        super(TrackNetV2DataLoader, self).__init__(batch_size)
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


@gin.configurable('tracknetv2_collate_fn')
def collate_fn(samples):
    batch_input_lengths = np.asarray(
        [x[0]['input_lengths'] for x in samples],
        dtype=np.int32
    )
    max_len = batch_input_lengths.max()
    batch_size = len(samples)
    # prepare batch vars
    batch_inputs = np.zeros((batch_size, max_len, samples[0][0]['inputs'].shape[1]), dtype=np.float32)
    batch_targets = np.zeros((batch_size, max_len, samples[0][1][0].shape[1]), dtype=np.float32)
    batch_target_masks = np.zeros((batch_size, max_len), dtype=np.bool)
    batch_input_masks = np.ones((batch_size, max_len), dtype=np.bool)
    # collect batches
    for i in range(batch_size):
        batch_inputs[i, :batch_input_lengths[i]] = samples[i][0]['inputs']
        batch_input_masks[i, :batch_input_lengths[i]] = samples[i][0]['mask']
        batch_targets[i, :batch_input_lengths[i]] = samples[i][1][0]
        batch_target_masks[i, :batch_input_lengths[i]] = samples[i][1][1]
    return {
        'inputs': torch.from_numpy(batch_inputs),
        'input_lengths': torch.from_numpy(batch_input_lengths),
        'mask': torch.from_numpy(batch_input_masks)
    }, (torch.from_numpy(batch_targets), torch.from_numpy(batch_target_masks))
