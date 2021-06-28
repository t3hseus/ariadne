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
from ariadne.utils import *
from ariadne.tracknet_v2_1.processor import TrackNetV21Processor, ProcessedTracknetDataChunk, ProcessedTracknetData, TracknetDataChunk

LOGGER = logging.getLogger('ariadne.prepare')

@gin.configurable(denylist=['data_df'])
class TrackNetV21ProcessorWithModel(TrackNetV21Processor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 valid_size: float,
                 device: str,
                 tracknet_v2_model: TrackNETv2,
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
        #self.model = tracknet_v2_model
        self.model = TrackNETv2(input_features=3)
        if tracknet_v2_checkpoint and os.path.isfile(tracknet_v2_checkpoint):
            if not torch.cuda.is_available():
                 self.model = weights_update(model=self.model, checkpoint=torch.load(tracknet_v2_checkpoint, map_location=torch.device('cpu')))
                 self.device = torch.device('cpu')
            else:
                self.model = weights_update(model=self.model, checkpoint=torch.load(tracknet_v2_checkpoint))
        self.model.to(self.device)
        self.model.eval()

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
            grouped_df = df[df['track'] != -1].groupby('track')
            last_station = df[df['station'] > 1][['z', 'phi']].values
            multiplicity = grouped_df.ngroups
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            chunk_data_x, chunk_data_y, chunk_data_real = brute_force_hits_two_first_stations(df)
            chunk_data_len = np.full(len(chunk_data_x), 2)
            chunk_data_event = np.full(len(chunk_data_x), chunk.id)

            LOGGER.info(f'=====> id {chunk.id}')
            chunk_prediction, chunk_gru = self.model(torch.tensor(chunk_data_x).to(self.device),
                                                      torch.tensor(chunk_data_len, dtype=torch.int64).to(self.device),
                                                      return_gru_states=True)
            chunk_prediction = chunk_prediction[:, -1, :]
            chunk_gru = chunk_gru[:, -1, :]
            last_station_index = store_in_index(np.ascontiguousarray(last_station), num_components=2)
            nearest_hits_index = search_in_index(chunk_prediction.detach().cpu().numpy()[:, :2],
                                                 last_station_index,
                                                 10,
                                                 n_dim=2)
            nearest_hits = last_station[nearest_hits_index]
            nearest_hits, in_ellipse = filter_hits_in_ellipses(chunk_prediction.cpu().detach().numpy(),
                                                                        nearest_hits,
                                                                        nearest_hits_index,
                                                                        filter_station=True,
                                                                        z_last=True,
                                                                        find_n=nearest_hits_index.shape[1],
                                                                        n_dim=2)

            is_close = np.all(np.isclose(nearest_hits, np.expand_dims(chunk_data_y, 1).repeat(nearest_hits.shape[1], axis=1)), axis=-1)
            found_real_track_ending = is_close & np.expand_dims(chunk_data_real.astype(bool), 1).repeat(is_close.shape[1], axis=1)
            is_close = np.all(np.isclose(nearest_hits, np.expand_dims(chunk_data_y, 1).repeat(nearest_hits.shape[1], 1)), -1)
    
            to_use = found_real_track_ending | in_ellipse
            chunk_gru = np.expand_dims(chunk_gru.detach().cpu().numpy(), 1).repeat(nearest_hits.shape[1], 1)
            nearest_hits = nearest_hits.reshape(-1, nearest_hits.shape[-1])
            nothing_for_real_track = ~np.any(to_use, axis=-1) & (chunk_data_real > 0.5) # if real track has nothing in ellipse or real last point is too far for faiss
            #chunk_gru_to_add = chunk_gru[nothing_for_real_track]
            to_use = to_use.reshape(-1)
            found_real_track_ending = is_close.reshape(-1)
            chunk_gru = chunk_gru.reshape(-1, chunk_gru.shape[-1])
            #chunk_gru = np.concatenate((chunk_gru, chunk_gru_to_add), axis=0)
            #nearest_hits = np.concatenate((nearest_hits, chunk_data_y_to_add), axis=0)
            #label = np.concatenate((is_close, np.ones(chunk_data_y_to_add.shape[0])), axis=0)
            #to_use = np.concatenate((to_use, np.ones(chunk_data_y_to_add.shape[0])), axis=0)
            #chunk_data_event = np.concatenate((chunk_data_event, np.full(chunk_data_y_to_add.shape[0], chunk_data_event[0])), axis=0)
            chunk_data_len = np.full(len(nearest_hits), 2)
            chunk_data_event = np.full(len(nearest_hits), chunk.id)
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
