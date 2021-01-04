from typing import Collection

import gin

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ariadne.point_net.point.points import load_points, Points


@gin.configurable
class PointsDatasetMemory(Dataset):

    def __init__(self, input_dir, n_samples=None):
        self.input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.npz')]
        self.filenames = (filenames[:n_samples] if n_samples is not None else filenames)
        self.points = {index: load_points(self.filenames[index])
                       for index in range(len(self.filenames))}

    def __getitem__(self, index):
        # uncomment to find bad events in the dataset:
        # print(index, self.filenames[index])
        return self.points[index]

    def __len__(self):
        return len(self.filenames)


@gin.configurable('points_collate_fn')
def collate_fn(points: Collection[Points]):
    batch_size = len(points)
    n_feat = 3

    n_dim = np.array([(p.X.shape[1]) for p in points])
    # print(n_dim)
    max_dim = n_dim.max()
    batch_inputs = np.zeros((batch_size, n_feat, max_dim), dtype=np.float32)
    batch_targets = np.zeros((batch_size, max_dim), dtype=np.float32)

    for i, p in enumerate(points):
        batch_inputs[i, :, :n_dim[i]] = p.X

        batch_targets[i, :n_dim[i]] = p.track

    return {"x": torch.from_numpy(batch_inputs)}, torch.from_numpy(batch_targets)
