from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gin
from ariadne.tracknet_v2_1.dataset import TrackNetV21Dataset, TrackNetClassifierDataset
from ariadne.utils.base import find_nearest_hit

import warnings
warnings.filterwarnings("ignore")


@gin.configurable    
class TrackNetV22Dataset(TrackNetV21Dataset):
    """TrackNET_v2_2 dataset with use of base TrackNetV2 pretrained model.
    It needs prepared data without model use (processor.py)

    Input features for classifier are created using first n-1 (now 2)
    coordinates of track-candidate and found hit.
    """

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
                is_close = np.all(np.isclose(found_hit, sample_y))
                if is_close:
                    found_real_track_ending = 1
        found_real_track_ending = torch.tensor(found_real_track_ending)
        if found_hit is None:
            found_hit = np.array([[-2., -2.]])
        temp_hit = np.full((1, 3), 0.9)
        temp_hit[0][1] = found_hit[0][0]
        temp_hit[0][2] = found_hit[0][1]
        track = np.concatenate([sample_inputs, temp_hit]).flatten()
        return [{'coord_features': torch.tensor(track).to(torch.float).detach()},
                found_real_track_ending.to(torch.float32)]

@gin.configurable
class TrackNetV22ClassifierDataset(TrackNetClassifierDataset):
    """TrackNET_v2_2 dataset without use of base TrackNetV2 pretrained model.
    It reads previously filtered data.
    It needs prepared data with model use (processor_with_model.py)

    Input features for classifier are created using first n-1 (now 2)
    coordinates of track-candidate and found hit.
    """

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_input = self.data['inputs'][idx]
        sample_pred = self.data['preds'][idx].flatten()
        sample_label = self.data['labels'][idx]
        temp_hit = np.full((1, 3), 0.9)
        temp_hit[0][1] = sample_pred[0]
        temp_hit[0][2] = sample_pred[1]
        track = np.concatenate([sample_input, temp_hit]).flatten()
        return [{'coord_features': torch.tensor(track).to(torch.float).detach()},
                torch.tensor(sample_label).to(torch.float32)]