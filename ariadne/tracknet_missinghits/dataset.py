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
class TrackNetDatasetWithMaskMissingHits(Dataset):
    """TrackNET_v2 dataset.

    Returns a pairs of dict and tuple:
    {'inputs': inputs, 'input_lengths': input_lengths}, (target, mask)
    inputs = track[:-1] # except the last timestep
    target = track[1:] # except the first timestep

    """
    def __init__(self,
                 data_path,
                 input_features=2,
                 mask_first_n_steps=-1,
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
        self.input_features = input_features
        if n_samples is not None:
            self.tracks = self.tracks[:n_samples]
        self.add_next_z_feat = add_next_z_feat
        self.use_index = use_index
        assert min_track_len > 2, "Track cannot be less than three hits"
        self.min_track_len = min_track_len
        self._mask_first_n_steps = mask_first_n_steps

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        track = self.tracks[idx]
        # except last timestep and (x, y, z) coords
        input_sample = track[:-1, :3]
        # r = sqrt(x^2 + y^2)
        #r_feature = np.sqrt(input_sample[:, 0]**2 + input_sample[:, 1]**2).reshape(-1, 1) + 0.02
        #input_sample = np.append(input_sample, r_feature, axis=1)
        
        input_dict = {
            'inputs': input_sample,
            'input_lengths': len(input_sample)
        }
        # except first timestep and only x and y coords
        target = self.tracks[idx][1:, :self.input_features]
        target = np.append(target, [[0, 0, 0]], axis=0)

        mask_target_1 = np.ones(input_dict['input_lengths']).astype(np.bool)
        mask_target_2 = np.ones(input_dict['input_lengths']).astype(np.bool)
        mask_target_2[-1] = False
        
        if self.use_index:
            return input_dict, (target, (mask_target_1, mask_target_2)), idx
        # else
        return input_dict, (target, (mask_target_1, mask_target_2))

    
@gin.configurable
class TrackNetDatasetWithMaskMissingHitsMissFeature(Dataset):
    """TrackNET_v2 dataset.

    Returns a pairs of dict and tuple:
    {'inputs': inputs, 'input_lengths': input_lengths}, (target, mask)
    inputs = track[:-1] # except the last timestep
    target = track[1:] # except the first timestep

    """
    def __init__(self,
                 data_path,
                 input_features=2,
                 mask_first_n_steps=-1,
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
        self.input_features = input_features
        if n_samples is not None:
            self.tracks = self.tracks[:n_samples]
        self.add_next_z_feat = add_next_z_feat
        self.use_index = use_index
        assert min_track_len > 2, "Track cannot be less than three hits"
        self.min_track_len = min_track_len
        self._mask_first_n_steps = mask_first_n_steps

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        track = self.tracks[idx]
        # except last timestep and (x, y, z) coords
        input_sample = track[:-1, :3]
        # miss feature - 0 if missed, 1 if hit
        miss_feature = (input_sample[:, 2] != 0).astype('float').reshape(-1, 1)
        input_sample = np.append(input_sample, miss_feature, axis=1)
        input_dict = {
            'inputs': input_sample,
            'input_lengths': len(input_sample)
        }
        # except first timestep
        target = input_sample[1:]
        target = np.append(target, [[0, 0, 0, 0]], axis=0)

        mask_target_1 = target[:, -1].astype(np.bool)
        mask_target_2 = np.zeros(input_dict['input_lengths']).astype(np.bool)
        mask_target_2[:-1] = target[1:, -1].astype(np.bool)
        
        target = target[:, :3]
        
        if self.use_index:
            return input_dict, (target, (mask_target_1, mask_target_2)), idx
        # else
        return input_dict, (target, (mask_target_1, mask_target_2))