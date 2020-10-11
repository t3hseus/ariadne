from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gin


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

@gin.configurable
class TrackNetV2Dataset(Dataset):
    """TrackNET_v2 dataset."""

    def __init__(self, input_dir, use_index=False, n_samples=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            preprocessing (callable, optional): Optional transform to be applied
                on a dataframe.
        """
        self.input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.npz')]
        self.filenames = (filenames[:n_samples] if n_samples is not None else filenames)
        d = {}
        self.all_data = {}
        for i, s in enumerate(self.filenames):
            with np.load(s) as data:
                if i == 0:
                    for k in data.files:
                        d[k] = []
                # Loop over all the keys
                for k in data.files:
                    d[k].append(data[k])
            # Merge arrays via np.vstack()
        for k, v in d.items():
            vv = np.concatenate(v, axis=0)
            self.all_data[k] = vv

        self.use_index = use_index
        self.data = dict()
        if n_samples is not None:
            for key in self.all_data.keys():
                print(key)
                self.data[key] = self.all_data[key][:n_samples]
        else:
            for key in self.all_data.keys():
                self.data[key] = self.all_data[key]


    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx][:2]
        sample_idx = idx
        if self.use_index:
            return {'inputs': sample_inputs, 'input_lengths': sample_len, 'y': sample_y}, sample_idx
        else:
            return {'inputs': sample_inputs, 'input_lengths': sample_len, 'y': sample_y}

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
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        sample_moment = self.data['moments'][idx]
        is_track = self.data['is_real'][idx]
        sample_idx = idx
        return {'x': {'inputs': sample_inputs, 'input_lengths': sample_len},
                'y': sample_y, 'index': sample_idx, 'moment': sample_moment, 'is_real_track': is_track}