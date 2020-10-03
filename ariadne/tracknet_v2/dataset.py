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

class PreprocessingBESDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, preprocessing=None, needed_columns=('r', 'phi', 'z'), use_next_z=False):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            preprocessing (callable, optional): Optional transform to be applied
                on a dataframe.
        """
        self.frame = pd.read_csv(csv_file)
        if preprocessing:
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


class TrackNetV2Dataset(Dataset):
    """TrackNET_v2 dataset."""

    def __init__(self, data_file, use_index=False):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            preprocessing (callable, optional): Optional transform to be applied
                on a dataframe.
        """
        self.data = np.load(data_file)
        self.use_index = use_index

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        sample_idx = idx
        if self.use_index:
            return {'x': {'inputs': sample_inputs, 'input_lengths': sample_len}, 'y': sample_y, 'index': sample_idx}
        else:
            return {'x': {'inputs': sample_inputs, 'input_lengths': sample_len}, 'y': sample_y}

class TrackNetV2ExplicitDataset(Dataset):
    """TrackNET_v2 dataset."""

    def __init__(self, data_file):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            preprocessing (callable, optional): Optional transform to be applied
                on a dataframe.
        """
        self.data = np.load(data_file)

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        sample_moment = self.data['moment'][idx]
        is_track = self.data['is_track'][idx]
        sample_idx = idx
        return {'x': {'inputs': sample_inputs, 'input_lengths': sample_len},
                'y': sample_y, 'index': sample_idx, 'moment': sample_moment, 'is_real_track': is_track}