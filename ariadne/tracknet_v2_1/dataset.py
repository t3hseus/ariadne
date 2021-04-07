from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gin
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.dataset import TrackNetV2Dataset
from ariadne.tracknet_v2.metrics import point_in_ellipse
from ariadne.utils import weights_update, find_nearest_hit, load_data
import warnings
warnings.filterwarnings("ignore")


@gin.configurable    
class TrackNetV21Dataset(TrackNetV2Dataset):
    """TrackNET_v2_1 dataset with use of base TrackNetV2 pretrained model.
    It needs prepared data without model use (processor.py)

    Input features are created using state of
    GRU-layer of tracknet and found hit.
    """

    def __init__(self,
                 path_to_checkpoint,
                 input_dir,
                 file_mask,
                 last_station_file_mask,
                 use_index=False,
                 n_samples=None,
                 model_input_features=4):
        super().__init__(input_dir=input_dir,
                         file_mask=file_mask,
                         use_index=use_index,
                         n_samples=n_samples)
        self.model = TrackNETv2(input_features=model_input_features)
        if path_to_checkpoint is not None:
            self.model = weights_update(model=self.model, checkpoint=torch.load(path_to_checkpoint))
        self.model.eval()
        self.use_index = use_index
        self.data = load_data(input_dir, file_mask, n_samples)
        all_last_station_data = load_data(input_dir, last_station_file_mask, n_samples)
        last_station_hits = all_last_station_data['hits']
        last_station_events = all_last_station_data['events']
        self.last_station_hits = torch.from_numpy(last_station_hits)
        self.last_station_events = torch.from_numpy(last_station_events)

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_inputs = self.data['inputs'][idx]
        sample_len = self.data['input_lengths'][idx]
        sample_y = self.data['y'][idx]
        is_track = self.data['is_real'][idx]

        sample_event = self.data['events'][idx]
        sample_prediction, sample_gru = self.model(torch.tensor(sample_inputs).unsqueeze(0),
                                                   torch.tensor([sample_len], dtype=torch.int64), return_gru_state=True)
        last_station_this_event = self.last_station_hits[self.last_station_events == sample_event.item()]
        nearest_hit, in_ellipse = find_nearest_hit(sample_prediction.detach().cpu().numpy(),
                                                   last_station_this_event.detach().cpu().numpy())
        found_hit = nearest_hit if in_ellipse else None
        found_real_track_ending = 0
        if found_hit is not None:
            if is_track:
                is_close = np.all(np.isclose(found_hit.flatten(), sample_y.flatten()))
                if is_close:
                    found_real_track_ending = 1
        found_real_track_ending = torch.tensor(found_real_track_ending)
        if found_hit is None:
            found_hit = np.array([-2., -2.])
        return [{'gru_features': sample_gru.detach().squeeze(),
                'coord_features': torch.tensor(found_hit).to(torch.double).detach().squeeze()},
                found_real_track_ending.long()]


@gin.configurable
class TrackNetClassifierDataset(TrackNetV2Dataset):
    """TrackNET_v2_1 dataset without use of base TrackNetV2 pretrained model.
    It needs prepared data using model (processor_with_model.py). For with dataset last station hits are not needed"""

    def __init__(self,
                 input_dir,
                 file_mask,
                 use_index=False,
                 n_samples=None,
                 is_train=True):
        super().__init__(input_dir=input_dir,
                         file_mask=file_mask,
                         use_index=use_index,
                         n_samples=n_samples)
        self.use_index = use_index
        self.is_train = is_train
        self.data = load_data(input_dir, file_mask, n_samples)
        if self.is_train:
            print(self.data['labels'])

    def __len__(self):
        return len(self.data['is_real'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_gru = self.data['grus'][idx]
        found_hit = self.data['preds'][idx]
        is_track = self.data['labels'][idx]
        if self.is_train:
            print(is_track)
        return [{'gru_features': torch.tensor(sample_gru).squeeze(),
                'coord_features': torch.tensor(found_hit).squeeze()},
                is_track.astype(int)]


