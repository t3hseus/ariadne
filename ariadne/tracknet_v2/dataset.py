from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

stations_z = [0, 0.5, 1]

stations_sizes = [[0.499827841387305,
                   0.4999727262294332,
                   0.99965568277461,
                   0.6278962737181447],
                  [0.50005704848717,
                    0.5001411506178041,
                    0.9998859030256596,
                    0.8142050534164471],
                  [0.5001180262696687,
                    0.5000589751012908,
                    0.9996381603027529,
                    0.9998796884320015]]

class BESDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, preprocessing=None, needed_columns=('r', 'phi', 'z'), use_next_z=False):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            preprocessing (callable, optional): Optional transform to be applied
                on a dataframe.
        """
        self.frame = pd.read_csv(csv_file)
        self.frame = preprocessing(self.frame)
        assert all([item in (list(self.frame.columns)) for item in needed_columns]), 'One or more columns are not in dataframe!'
        self.grouped_frame = deepcopy(self.frame)
        self.grouped_frame = self.frame.groupby(by=['event', 'track'], as_index=False)
        self.group_keys = list(self.grouped_frame.groups.keys())
        self.needed_columns = needed_columns
        self.use_next_z = use_next_z

    def __len__(self):
        return self.grouped_frame.n_groups

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.grouped_frame.get_group(self.group_keys[idx])
        sample.reset_index()
        sample = sample.loc[:, self.needed_columns]
        y = deepcopy(sample.iloc[2, :].values)
        if self.use_next_z:
            for i in range(2):
                sample.loc[i, 'z+1'] = sample[i+1]['z']
        return {'x': {'inputs': sample[:2].values, 'input_lengths': 2}, 'y': y}

