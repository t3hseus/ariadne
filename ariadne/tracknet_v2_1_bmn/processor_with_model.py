import logging
import itertools
import os
from typing import List, Tuple, Optional, Iterable
from copy import deepcopy

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import gin

from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.metrics import point_in_ellipse
from ariadne.preprocessing import (
    BaseTransformer,
    DataProcessor,
    DataChunk,
    ProcessedDataChunk,
    ProcessedData
)
from ariadne.utils import brute_force_hits_two_first_stations_cart, \
    find_nearest_hit, \
    weights_update, \
    store_in_index,\
    filter_hits_in_ellipse,\
    search_in_index, \
    is_ellipse_intersects_station

from ariadne.tracknet_v2_1.processor import TrackNetV21Processor, ProcessedTracknetDataChunk, ProcessedTracknetData, TracknetDataChunk
from ariadne.tracknet_v2_bmn.processor_bmn import TrackNetBMNProcessor

LOGGER = logging.getLogger('ariadne.prepare')

@gin.configurable(denylist=['data_df'])
class TrackNetV21BMNProcessorWithModel(TrackNetV21Processor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 valid_size: float,
                 device: str,
                 tracknet_v2_model: TrackNETv2,
                 num_stations: int =6,
                 n_times_oversampling: int = 4,
                 tracknet_v2_checkpoint: str = '',
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            data_df=data_df,
            output_dir=output_dir,
            transforms=transforms,
            name_suffix=name_suffix,
            valid_size=valid_size,
            n_times_oversampling=n_times_oversampling
            )
        self.output_name = f'{self.output_dir}/{name_suffix}'
        self.n_times_oversampling = n_times_oversampling
        self.valid_size = valid_size
        self.chunks = []
        self.device = torch.device(device)
        self.max_n_stations = num_stations
        #self.model = tracknet_v2_model
        self.z_values = {0: -0.9998595, 1: -0.953343, 2: -0.841152, 3: -0.675508, 4: -0.379517, 5: 0.180436}
        self.model = TrackNETv2(input_features=3, z_values=self.z_values)
        if tracknet_v2_checkpoint and os.path.isfile(tracknet_v2_checkpoint):
            self.model = weights_update(model=self.model, checkpoint=torch.load(tracknet_v2_checkpoint))
        self.model.eval()

        self.stations_sizes = np.array([
                                [[0.5390, 15.99873, 163.2, 45]],
                                [[0.7025, 16.20573, 163.2, 45]],
                                [[1.9925, 16.36073, 163.2, 45]],
                                # module 0, module 1
                                [[3.0860, 16.40473, 163.2, 45]],
                                [[3.7980, 16.09373, 163.2, 45]],
                                [[4.5815, 16.45473, 163.2, 45]]])

        constraints = {'x': [-81.03, 86.15], 'y': [-17.1, 39.0], 'z': [11.97, 183.82]}
        self.stations_sizes[:, :, 1] = 2 * (self.stations_sizes[:,:,  1] - constraints['x'][0]) / \
                                    (constraints['x'][1] - constraints['x'][0]) - 1
        self.stations_sizes[:, :,  3] = 2 * (self.stations_sizes[:, :, 3] - constraints['x'][0]) / \
                                    (constraints['x'][1] - constraints['x'][0]) - 1
        self.stations_sizes[:, :,  0] = 2 * (self.stations_sizes[:, :, 0] - constraints['x'][0]) / \
                                    (constraints['y'][1] - constraints['y'][0]) - 1
        self.stations_sizes[:, :,  2] = 2 * (self.stations_sizes[:, :, 2] - constraints['x'][0]) / \
                                    (constraints['y'][1] - constraints['y'][0]) - 1
        self.det_indices = [0, 1]

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        if len(self.det_indices) > 1:
            chunk_df.loc[chunk_df.det == 1, 'station'] = chunk_df.loc[chunk_df.det == 1, 'station'].values + 3

        tracks = chunk_df[chunk_df['track'] != -1].groupby('track')
        for i, track in tracks :
            print(track)
        processed = self.transformer(chunk_df)
        processed.loc[processed['station'] > self.max_n_stations, 'bucket'] = -1
        tracks = processed[processed['track'] != -1].groupby('track')
        for i, track in tracks:
            print(track)
        self.z_values = np.sort(processed['z'].unique())
        return TracknetDataChunk(processed[processed['bucket']!=-1])

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        chunk_df = chunk.df_chunk_data
        if chunk_df.empty:
            return ProcessedTracknetDataChunk(None, '', -1)
        chunk_id = int(chunk_df.event.values[0])
        output_name = f'{self.output_dir}/tracknet_with_model_{idx.replace(".txt", "")}'
        self.chunks.append(chunk_id)
        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            if chunk.processed_object is None:
                continue
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_momentum = []
            df = chunk.processed_object
            print(df)
            grouped_df = df[df['track'] != -1].groupby('track')
            tracks = df[df['track'] != -1].groupby('track')
            for i, track in grouped_df:
                print(track)
            track_list = []
            print(self.z_values)
            return
            for i, track in tracks:
                if len(track) > self.max_n_stations:
                    continue
                temp_track = np.zeros((self.max_n_stations, 3), 'float')
                temp_track[:len(track)] = track[['z', 'x', 'y']].values
                track_list.append(temp_track)
            track_list = np.stack(track_list, 0)
            multiplicity = grouped_df.ngroups
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            chunk_data_x, chunk_data_y, chunk_data_real = brute_force_hits_two_first_stations_cart(df)  # initial seeds
            chunk_data_len = np.full(len(chunk_data_x), 2)
            chunk_data_event = np.full(len(chunk_data_x), chunk.id)
            event_hits = np.ascontiguousarray(df[['z', 'x', 'y']].values)
            event_hits[:, 0] *= 100.
            event_index = store_in_index(event_hits)  # z is first!
            LOGGER.info(f'=====> id {chunk.id}')
            j = 2
            track_candidates = []
            gru_candidates = []
            track_labels = []
            for i in range(2, self.max_n_stations+1):
                flag = 0
                print(i)
                chunk_prediction, chunk_gru = self.model(torch.tensor(chunk_data_x).to(self.device),
                                                          torch.tensor(chunk_data_len, dtype=torch.int64).to(self.device),
                                                          return_gru_state=True)
                new_pred = torch.full((len(chunk_prediction), 1), self.z_values[i]*100.)
                chunk_prediction = torch.cat((new_pred, chunk_prediction), 1)  # adding z-value to predicted ellipses
                nearest_hits_index = search_in_index(chunk_prediction.detach().cpu().numpy()[:, :3], event_index, 10)
                for el_num, ellipse in enumerate(chunk_prediction):
                    nearest_this_ellipse = nearest_hits_index[el_num]
                    nearest_this_ellipse = nearest_this_ellipse[nearest_this_ellipse != -1]
                    found_hits = event_hits[nearest_this_ellipse]
                    nearest_this_ellipse, in_ellipse = filter_hits_in_ellipse(ellipse.cpu().detach().numpy(), found_hits, filter_station=True, z_last=False)
                    print(nearest_this_ellipse)
                    if len(nearest_this_ellipse) > 0:
                        nearest_this_ellipse[:, 0] /= 100.
                        extensions = np.zeros((len(nearest_this_ellipse), i+1, 3))
                        extensions[:, :i] = chunk_data_x[el_num]
                        extensions[:, i] = nearest_this_ellipse
                        if el_num == 0:
                            batch_xs_new = extensions
                        elif batch_xs_new.shape[1] < i+1:
                            batch_xs_new = extensions
                        else:
                            batch_xs_new = np.concatenate((batch_xs_new, extensions), axis=0)
                        if i == 5:
                            for cand in batch_xs_new:
                                label = np.any(np.all(np.all(np.equal(track_list[:, :i],
                                                                      np.expand_dims(cand, 0).repeat(len(track_list), 0)),
                                                             axis=-1),axis=-1))
                                print(label)
                                gru_candidates.extend(chunk_gru[el_num])
                                track_labels.append(label)
                    else:
                        if i > 2 and i<5:
                            is_ellipse_on_station = is_ellipse_intersects_station(ellipse[1:], self.stations_sizes[i])
                            if not is_ellipse_on_station:
                                 track_candidates.extend(chunk_data_x[el_num])
                                 label = False
                                 for track in track_list:
                                     label = np.any(np.all(np.all(np.equal(track_list[:, :i],
                                                                         np.expand_dims(chunk_data_x[el_num], 0).repeat(len(track_list), 0)), axis=-1),
                                                         axis=-1))
                                     print(label)
                                 gru_candidates.extend(chunk_gru[el_num])
                                 track_labels.append(label)
                            else:
                                continue
                        else:
                            continue
                print(batch_xs_new)
                chunk_data_x = batch_xs_new
                chunk_data_len = np.full(len(chunk_data_x), i+1)
                j += 1
            print(np.array(track_labels).sum())
            print()
                #nearest_hits = nearest_hits.reshape(-1, nearest_hits.shape[-1])
                #nothing_for_real_track = ~np.any(to_use, axis=-1) & (chunk_data_real > 0.5) # if real track has nothing in ellipse or real last point is too far for faiss
                #chunk_gru_to_add = chunk_gru[nothing_for_real_track]
                # chunk_data_y_to_add = chunk_data_y[nothing_for_real_track]
                #to_use = to_use.reshape(-1)
                #found_real_track_ending = is_close.reshape(-1)
                #chunk_gru = chunk_gru.reshape(-1, chunk_gru.shape[-2], chunk_gru.shape[-1])
                #chunk_gru = np.concatenate((chunk_gru, chunk_gru_to_add), axis=0)
                #nearest_hits = np.concatenate((nearest_hits, chunk_data_y_to_add), axis=0)
                #label = np.concatenate((is_close, np.ones(chunk_data_y_to_add.shape[0])), axis=0)
                #to_use = np.concatenate((to_use, np.ones(chunk_data_y_to_add.shape[0])), axis=0)
                #chunk_data_event = np.concatenate((chunk_data_event, np.full(chunk_data_y_to_add.shape[0], chunk_data_event[0])), axis=0)
                #chunk_data_len = np.full(len(nearest_hits), 2)
                #chunk_data_event = np.full(len(nearest_hits), chunk.id)
            chunk_data = {'x': {'grus': chunk_gru[to_use], 'preds': nearest_hits[to_use]},
                          'label': found_real_track_ending[to_use],
                          'event': chunk_data_event[to_use],
                          'multiplicity': multiplicity}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks, chunks[0].output_name)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        train_data_preds = []
        train_data_inputs = []
        train_data_labels = []
        train_data_events = []

        valid_data_preds = []
        valid_data_inputs = []
        valid_data_labels = []
        valid_data_events = []

        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.id in train_chunks:
                initial_len = len(data_chunk.processed_object['label'])
                multiplicity = data_chunk.processed_object['multiplicity']
                max_len = int(multiplicity + (initial_len / self.n_times_oversampling))
                to_use = data_chunk.processed_object['label'] | (np.arange(initial_len) < max_len)

                train_data_inputs.append(data_chunk.processed_object['x']['grus'][to_use])
                train_data_preds.append(data_chunk.processed_object['x']['preds'][to_use])
                train_data_labels.append(data_chunk.processed_object['label'][to_use])
                train_data_events.append(data_chunk.processed_object['event'][to_use])
            else:
                valid_data_inputs.append(data_chunk.processed_object['x']['grus'])
                valid_data_preds.append(data_chunk.processed_object['x']['preds'])
                valid_data_labels.append(data_chunk.processed_object['label'])
                valid_data_events.append(data_chunk.processed_object['event'])

        train_data_inputs = np.concatenate(train_data_inputs)
        train_data_preds = np.concatenate(train_data_preds)
        train_data_labels = np.concatenate(train_data_labels)
        train_data_events = np.concatenate(train_data_events)

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_preds = np.concatenate(valid_data_preds)
        valid_data_labels = np.concatenate(valid_data_labels)
        valid_data_events = np.concatenate(valid_data_events)

        np.savez(
            f'{processed_data.output_name}_train',
            grus=train_data_inputs,
            preds=train_data_preds,
            labels=train_data_labels,  # predicted right point and point was real
            events=train_data_events
        )
        np.savez(
            f'{processed_data.output_name}_valid',
            grus=valid_data_inputs,
            preds=valid_data_preds,
            labels=valid_data_labels,
            events=valid_data_events
        )
        LOGGER.info(f'Saved train hits to: {processed_data.output_name}_train.npz')
        LOGGER.info(f'Saved valid hits to: {processed_data.output_name}_valid.npz')
