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
            self.model.eval()
        self.input_dir = os.path.expandvars(input_dir)
        self.use_index = use_index
        self.data = self.load_data(input_file, n_samples)
        filename = os.path.join(self.input_dir, last_station_file)
        all_last_station_coordinates = np.load(filename)
        last_station_hits = all_last_station_coordinates['hits'][:, 1:]
        #print(self.data['y'].shape)
        #print(self.data['events'].shape)
        self.last_station_hits = torch.from_numpy(self.data['y'][:, 1:])
        self.last_station_events = torch.from_numpy(self.data['events'])




    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.data.keys())
        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx][1:]
        #print(sample_y.shape)
        sample_moment = self.data['moments'][idx]
        is_track = self.data['is_real'][idx]
        sample_idx = idx
        sample_event = self.data['events'][idx]
        sample_prediction = self.model(torch.tensor(sample_inputs).unsqueeze(0),
                                       torch.tensor([sample_len], dtype=torch.int64))
        sample_gru = self.model.last_gru_output
        nearest_hit, in_ellipse = self.find_nearest_hit(sample_prediction, sample_event)
        found_hit = nearest_hit if in_ellipse == True else torch.tensor([[-2., -2.]], dtype=torch.float64)
        if found_hit is not None:
            if is_track:
                if found_hit == sample_y:
                    print('this is a real track')
                    found_real_track_ending = 0.0
                else:
                    found_real_track_ending = 1.0
            else:
                found_real_track_ending = 1.0
        else:
            print('found_nothing')
            found_real_track_ending = 1.0

        found_real_track_ending = torch.tensor(found_real_track_ending)
        return [{'gru_x': sample_gru.detach(),
                'coord_x': found_hit.detach()},
                found_real_track_ending]

    def load_data(self, input_file, n_samples):
        filename = os.path.join(self.input_dir, input_file)
        all_data = {}
        with np.load(filename) as data:
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
    
    def find_nearest_hit(self, ellipses, event_id):
        centers = ellipses[:, :2]
        last_station_hits = self.last_station_hits[self.last_station_events == event_id.item()]
        dists = torch.cdist(last_station_hits.float(), centers.float())
        minimal = self.last_station_hits[torch.argmin(dists, dim=0)]
        is_in_ellipse = point_in_ellipse(ellipses, minimal)
        return minimal, is_in_ellipse
