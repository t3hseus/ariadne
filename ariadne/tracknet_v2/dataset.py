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
    """TrackNET_v2 dataset.

    Contains first k hits of tracks as x and final hit as y. Can be used for train/test of TrackNETv2.
    """

    def __init__(self, input_dir='', file_mask='', use_index=False, n_samples=None):
        """
        Args:
            input_dir (string): Path to files with data.
            n_samples (int): Maximum number of samples, optional
            use_index (int)

        """
        self.data = load_data(input_dir, file_mask, n_samples)
        self.use_index = use_index


    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        sample_idx = idx
        if self.use_index:
            return {'inputs': sample_inputs, 'input_lengths': sample_len}, sample_y, sample_idx
        else:
            return {'inputs': sample_inputs, 'input_lengths': sample_len}, sample_y

@gin.configurable
class TrackNetV2ExplicitDataset(TrackNetV2Dataset):
    """Explicit TrackNET_v2 dataset.

    It contains not only coordinates of first k hits and ending of track,
    but index of hit, momentum of track and is it real track or synthetic.

    It can be used for classifier training or for testing.
    """

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        sample_momentum = self.data['momentums'][idx]
        is_track = self.data['is_real'][idx]
        sample_idx = idx
        return {'x': {'inputs': sample_inputs, 'input_lengths': sample_len},
                'y': sample_y, 'index': sample_idx, 'momentum': sample_momentum, 'is_real_track': is_track}



@gin.configurable
class TrackNetV2DatasetWithMask(Dataset):
    """TrackNET_v2 dataset.

    Returns a pairs of dict and tuple:
    {'inputs': inputs, 'input_lengths': input_lengths}, (target, mask)
    inputs = track[:-1] # except the last timestep
    target = track[1:] # except the first timestep

    """
    def __init__(self,
                 data_path,
                 min_track_len=3,
                 add_next_z_feat=False,
                 use_index=False,
                 n_samples=None):
        """
        Args:
            data_path (string): Path to file with tracks.
            min_track_len (int): From which hit start to track prediction
            add_next_z_feat (bool): additional feature to the data (x, y, z) -> (x, y, z, z+1)
            n_samples (int): Maximum number of samples, optional
            use_index (bool): whether or not return index of track

        """
        self.tracks = np.load(data_path, allow_pickle=True)
        if n_samples is not None:
            self.tracks = self.tracks[:n_samples]
        self.add_next_z_feat = add_next_z_feat
        self.use_index = use_index
        assert min_track_len > 2, "Track cannot be less than three hits"
        self.min_track_len = min_track_len
        self._mask_first_n_steps = self.min_track_len - 2

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # except last timestep and (x, y, z) coords
        input_sample = self.tracks[idx][:-1, :3]
        input_dict = {
            'inputs': input_sample,
            'input_lengths': len(input_sample)
        }
        # except first timestep and only x and y coords
        target = self.tracks[idx][1:, :2]
        if self.add_next_z_feat:
            input_dict['inputs'] = np.hstack([
                input_sample,
                self.tracks[idx][1:, 2:3]
            ])

        mask = np.ones(input_dict['input_lengths']).astype(np.bool)
        mask[:self._mask_first_n_steps] = True#False

        if self.use_index:
            return input_dict, (target, mask), idx
        # else
        return input_dict, (target, mask)