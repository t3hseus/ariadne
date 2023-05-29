from typing import Collection, Dict

import gin

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ariadne.point_net.point.points import load_points, Points


class MemoryDataset(Dataset):

    def __init__(self, input_dir):
        super(MemoryDataset, self).__init__()
        self.input_dir = input_dir

    def load_data(self):
        raise NotImplementedError


class ItemLengthGetter(object):
    def __init__(self):
        pass

    def get_item_length(self, item_index):
        raise NotImplementedError


@gin.configurable
class CloudDataset(MemoryDataset, ItemLengthGetter):

    def __init__(self, input_dir, n_samples=None, pin_mem=False, sample_len=2048):
        super().__init__(os.path.expandvars(input_dir))
        self.n_samples = n_samples
        self.filenames = []
        self.points = {}
        self.pin_mem = pin_mem
        self.sample_len = sample_len

        filenames = [
            os.path.join(self.input_dir, f) for f in
            os.listdir(self.input_dir) if f.endswith('.npz')
        ]
        self.filenames = (
            filenames[:self.n_samples] if self.n_samples is not None else filenames
        )

    def __getitem__(self, index):
        if self.pin_mem:
            if index in self.points:
                item = self.points[index]
            else:
                item = self.points[index] = load_points(self.filenames[index])

            return item
        return load_points(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


class SubsetWithItemLen(Subset, ItemLengthGetter):
    def __init__(self, dataset, indices):
        super(SubsetWithItemLen, self).__init__(dataset, indices)

    def get_item_length(self, item_index):
        return len(self.dataset[self.indices[item_index]].track)


@gin.configurable('cloud_collate_fn')
def collate_fn(points: Collection[Points], sample_len=512, shuffle=True):
    batch_size = len(points)
    n_feat = 3
    n_dim = np.array([(p.X.shape[1]) for p in points])
    max_dim = sample_len  # n_dim.max()
    batch_inputs = np.zeros((batch_size, n_feat + 2, max_dim), dtype=np.float32) - 9.
    batch_mask = np.zeros((batch_size, max_dim), dtype=np.float32)

    batch_targets = np.zeros((batch_size, max_dim), dtype=np.float32)
    for i, p in enumerate(points):
        batch_mask[i, :n_dim[i]] = 1.
        perm_ids = np.arange(0, p.X.shape[1])
        assert perm_ids[-1] == p.X.shape[1] - 1, 'You need to specify upper bound!'
        if shuffle:
            perm_ids = np.random.permutation(p.X.shape[1])
        batch_inputs[i, :-2, :n_dim[i]] = p.X[:, perm_ids]
        batch_targets[i, :n_dim[i]] = np.float32(p.track[perm_ids] != -1)

    # max_dim = 512
    # batch_inputs = batch_inputs[:, :, :max_dim]
    # batch_targets = batch_targets[:, :max_dim]
    # batch_mask = batch_mask[:, :max_dim]
    batch_inputs = np.swapaxes(batch_inputs, -1, -2)
    batch_targets = np.expand_dims(batch_targets, -1)

    x = torch.from_numpy(batch_inputs)
    all_dists = torch.cdist(x, x)
    dists = torch.kthvalue(all_dists, 2, dim=-1).values
    x[..., -2] = dists
    dists = torch.kthvalue(all_dists, 3, dim=-1).values
    x[..., -1] = dists

    return (
        {"x": x, "mask": torch.from_numpy(batch_mask)},
        {"y": torch.from_numpy(batch_targets),  "mask": torch.from_numpy(batch_mask)}
    )

