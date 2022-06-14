from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gin

from ariadne.utils.data import load_data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


@gin.configurable
class SelectorDataset(Dataset):
    """Selector dataset.

    Returns a pairs of dict and tuple:
    {'inputs': inputs, 'input_lengths': input_lengths}
    inputs = track
    target = y (label of track)

    """
    def __init__(self,
                 data_path,
                 use_index=False,
                 n_samples=None):
        """
        Args:
            data_path (string): Path to file with tracks.
            use_index (bool): whether or not return index of track
            n_samples (int): Maximum number of samples, optional

        """
        data = np.load(data_path, allow_pickle=True)
        self.tracks = data[0]
        self.labels = data[1]
        
        if n_samples is not None:
            self.tracks = self.tracks[:n_samples]
            self.labels = self.labels[:n_samples]
        self.use_index = use_index
        
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_sample = self.tracks[idx].squeeze()
        input_sample = input_sample[:, :13]
        input_dict = {
            'inputs': input_sample,
            'input_lengths': len(input_sample)
        }

        #target = np.expand_dims(np.array([float(self.labels[idx])], dtype='float32'), 1)
        
        target = np.zeros(2)
        target[self.labels[idx]] = 1.

        if self.use_index:
            return input_dict, target, idx
        # else
        return input_dict, target