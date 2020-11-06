from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gin
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.metrics import point_in_ellipse

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

    def __init__(self, input_dir, input_file='', use_index=False, n_samples=None):
        """
        Args:
            input_dir (string): Path to files with data.
            n_samples (int): Maximum number of samples, optional
            use_index (int)

        """
        self.input_dir = os.path.expandvars(input_dir)
        filename = os.path.join(self.input_dir, input_file)
        
        self.all_data = {}
        with np.load(filename) as data:
            for k in data.files:
                self.all_data[k] = data[k]
        self.use_index = use_index
        self.data = dict()
        if n_samples is not None:
            for key in self.all_data.keys():
                n_samples = min((self.all_data[key].size, n_samples))
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
        sample_y = self.data['y'][idx][1:]
        sample_idx = idx
        if self.use_index:
            return {'inputs': sample_inputs, 'input_lengths': sample_len}, sample_y, sample_idx
        else:
            return {'inputs': sample_inputs, 'input_lengths': sample_len}, sample_y

@gin.configurable    
class TrackNetV2_1Dataset(TrackNetV2Dataset):
    """TrackNET_v2 dataset."""

    def __init__(self, input_dir, path_to_checkpoint, input_file, last_station_file,
                 use_index=False, n_samples=None, model_input_features=4):
        super().__init__(input_dir, input_file=input_file, use_index=use_index, n_samples=n_samples)
        if path_to_checkpoint is not None:
            self.model = self.weights_update(model=TrackNETv2(input_features=model_input_features),
                                   checkpoint=torch.load(path_to_checkpoint))
        filename = os.path.join(self.input_dir, last_station_file)
        all_last_station_coordinates = np.load(filename)
        last_station_hits = all_last_station_coordinates['hits'][:, 1:]
        self.last_station_hits = torch.from_numpy(last_station_hits)

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx][1:]
        sample_moment = self.data['moments'][idx]
        is_track = self.data['is_real'][idx]
        sample_idx = idx
        sample_prediction = self.model(torch.tensor(sample_inputs).unsqueeze(0),
                                       torch.tensor([sample_len], dtype=torch.int64))
        sample_gru = self.model.last_gru_output
        nearest_hit, in_ellipse = self.find_nearest_hit(sample_prediction)

        found_hit = nearest_hit if in_ellipse == True else torch.tensor([[-2., -2.]], dtype=torch.float64)
        if found_hit is not None:
            if found_hit == sample_y:
                print('this is right')
                is_prediction_right = 0.0
            else:
                is_prediction_right = 1.0
        else:
            is_prediction_right = 2.0

        is_prediction_right = torch.tensor(is_prediction_right)
        return [{'gru_x': sample_gru.detach(),
                'coord_x': found_hit.detach()},
                is_prediction_right]

    def weights_update(self, model, checkpoint):
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        real_dict = {}
        for (k, v) in model_dict.items():
            needed_key = None
            for pretr_key in pretrained_dict:
                if k in pretr_key:
                    needed_key = pretr_key
                    break
            assert needed_key is not None, "key %s not in pretrained_dict %r!" % (k, pretrained_dict.keys())
            real_dict[k] = pretrained_dict[needed_key]

        model.load_state_dict(real_dict)
        model.eval()
        return model
    
    def find_nearest_hit(self, ellipses):
        centers = ellipses[:,:2]
        dists = torch.cdist(self.last_station_hits.float(), centers.float())
        min_argument = torch.argmin(dists, dim=0)
        minimal = self.last_station_hits[torch.argmin(dists, dim=0)]
        is_in_ellipse = point_in_ellipse(ellipses, minimal)
        return minimal, is_in_ellipse
