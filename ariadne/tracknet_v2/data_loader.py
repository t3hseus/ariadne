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
                 max_size: int,
                 n_valid: float,
                 with_random=True):
        super(TrackNetV2DataLoader, self).__init__(batch_size)
        self.dataset = dataset(n_samples=max_size)
        data_size = len(self.dataset)
        n_valid = int(data_size * n_valid)
        n_train = data_size - n_valid
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn)

def collate_fn(samples):
    """
    """
    #print(samples)
    batch_size = len(samples)
    # Special handling of batch size 1
    # Get the matrix sizes in this batch
    from collections import defaultdict
    res = defaultdict(list)
    {res[key].append(sub[key]) for sub in samples for key in sub}
    #print(res)
    samples = res
    inputs = samples['inputs']
    y = samples['y']
    input_lengths = samples['input_lengths']
    max_len = max(input_lengths)
    n_features = samples['inputs'][0].shape[-1]
    n_features_y = samples['y'][0].shape[-1]
    # Allocate the tensors for this batch
    batch_inputs = np.zeros((batch_size, max_len, n_features), dtype=np.float32)
    batch_y = np.zeros((batch_size, n_features_y), dtype=np.float32)
    # Loop over samples and fill the tensors
    for i, g in enumerate(samples['input_lengths']):
        batch_inputs[i, :input_lengths[i]] = inputs[i]
        batch_y[i] = y[i]
    batch_inputs = [torch.from_numpy(batch_inputs), torch.from_numpy(np.array(input_lengths))]
    batch_target = torch.from_numpy(batch_y)
    return batch_inputs, batch_target
