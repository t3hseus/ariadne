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
from ariadne.utils import weights_update, find_nearest_hit
import warnings
warnings.filterwarnings("ignore")

@gin.configurable
class TrackNetV2Dataset(Dataset):
    """TrackNET_v2 dataset."""
    def __init__(self, input_file='', use_index=False, n_samples=None):
        """
        Args:
            input_dir (string): Path to files with data.
            n_samples (int): Maximum number of samples, optional
            use_index (int)

        """
        self.data = self.load_data(input_file, n_samples)


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

    def load_data(self, input_file, n_samples):
        all_data = {}
        with np.load(input_file) as data:
            for k in data.files:
                all_data[k] = data[k]

        data = dict()
        if n_samples is not None:
            for key in all_data.keys():
                n_samples = min((all_data[key].size, n_samples))
                data[key] = all_data[key][:n_samples]
        else:
            for key in all_data.keys():
                data[key] = all_data[key]
        return data

@gin.configurable    
class TrackNetV21Dataset(TrackNetV2Dataset):
    """TrackNET_v2_1 dataset with use of base TrackNetV2 pretrained model.
    It needs prepared data without model use (processor.py)"""

    def __init__(self, path_to_checkpoint, input_file, last_station_file,
                 use_index=False, n_samples=None, model_input_features=4):
        super().__init__(input_file=input_file, use_index=use_index, n_samples=n_samples)
        if path_to_checkpoint is not None:
            self.model = weights_update(model=TrackNETv2(input_features=model_input_features),
                                   checkpoint=torch.load(path_to_checkpoint))
            self.model.eval()
        self.use_index = use_index
        self.data = self.load_data(input_file, n_samples)
        all_last_station_data = np.load(last_station_file)
        last_station_hits = all_last_station_data['hits']
        last_station_events = all_last_station_data['events']
        print(last_station_hits)
        self.last_station_hits = torch.from_numpy(last_station_hits)
        print(last_station_hits)
        self.last_station_events = torch.from_numpy(last_station_events)

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx][1:]
        sample_moment = self.data['moments'][idx]
        is_track = self.data['is_real'][idx]
        sample_event = self.data['events'][idx]
        sample_prediction = self.model(torch.tensor(sample_inputs).unsqueeze(0),
                                       torch.tensor([sample_len], dtype=torch.int64))
        sample_gru = self.model.last_gru_output
        last_station_this_event = self.last_station_hits[self.last_station_events == sample_event.item()]
        nearest_hit, in_ellipse = find_nearest_hit(sample_prediction, last_station_this_event)
        found_hit = nearest_hit if in_ellipse == True else None
        if found_hit is not None:
            if is_track:
                is_close = torch.isclose(found_hit.to(torch.float32), torch.tensor(sample_y, dtype=torch.float32))
                is_close = is_close.sum(dim=1) / 2.
                if is_close:
                    found_real_track_ending = 0
                else:
                    found_real_track_ending = 1
            else:
                found_real_track_ending = 1
        else:
            found_real_track_ending = 1

        found_real_track_ending = torch.tensor(found_real_track_ending)
        if found_hit is None:
            found_hit = torch.tensor([-2., -2.], dtype=torch.double)
        return [{'gru_x': sample_gru.detach().squeeze(),
                'coord_x': found_hit.detach().squeeze()},
                found_real_track_ending.long()]


@gin.configurable
class ClassifierDataset(TrackNetV2Dataset):
    """TrackNET_v2_1 dataset without use of base TrackNetV2 pretrained model.
    It needs prepared data using model (processor_with_model.py). For with dataset last station hits are not needed"""

    def __init__(self, input_file,use_index=False, n_samples=None):
        super().__init__( input_file=input_file, use_index=use_index, n_samples=n_samples)
        self.use_index = use_index
        self.data = self.load_data(input_file, n_samples)

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_gru = self.data['gru'][idx]
        sample_y = self.data['y'][idx]
        found_hit = self.data['preds'][idx]
        sample_moment = self.data['moments'][idx]
        is_track = self.data['is_real'][idx]
        sample_idx = idx
        sample_event = self.data['events'][idx]
        return [{'gru_x': torch.tensor(sample_gru).squeeze(),
                'coord_x': torch.tensor(found_hit).squeeze()},
                sample_y.astype(int)]


