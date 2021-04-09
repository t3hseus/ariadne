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
from ariadne.utils import brute_force_hits_two_first_stations, find_nearest_hit, weights_update
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
            self.model = weights_update(model=self.model, checkpoint=torch.load(tracknet_v2_checkpoint))
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
            last_station = df[df['station'] > 1][['phi', 'z']].values
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
                                                      return_gru_state=True)
            nearest_hits, in_ellipse = find_nearest_hit(chunk_prediction.detach().cpu().numpy(),
                                                        np.ascontiguousarray(last_station))
            is_close = np.all(np.isclose(nearest_hits, chunk_data_y), axis=1)
            found_real_track_ending = is_close & chunk_data_real.astype(bool)
            chunk_data = {'x': {'gru': chunk_gru[in_ellipse], 'preds': nearest_hits[in_ellipse]},
                          'label': found_real_track_ending[in_ellipse],
                          'is_real': chunk_data_real[in_ellipse],
                          'event': chunk_data_event[in_ellipse],
                          'multiplicity': multiplicity}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks, chunks[0].output_name)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        train_data_preds = []
        train_data_grus = []
        train_data_labels = []
        train_data_real = []
        train_data_events = []

        valid_data_preds = []
        valid_data_grus = []
        valid_data_labels = []
        valid_data_real = []
        valid_data_events = []

        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        valid_chunks = list(set(self.chunks) - set(train_chunks))
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.id in train_chunks:
                initial_len = len(data_chunk.processed_object['label'])
                multiplicity = data_chunk.processed_object['multiplicity']
                max_len = int(multiplicity + (initial_len / self.n_times_oversampling))
                train_data_grus.append(data_chunk.processed_object['x']['gru'][0:max_len].detach().cpu().numpy())
                train_data_preds.append(data_chunk.processed_object['x']['preds'][0:max_len])
                train_data_labels.append(data_chunk.processed_object['label'][0:max_len])
                train_data_real.append(data_chunk.processed_object['is_real'][0:max_len])
                train_data_events.append(data_chunk.processed_object['event'][0:max_len])
            else:
                valid_data_grus.append(data_chunk.processed_object['x']['gru'].detach().cpu().numpy())
                valid_data_preds.append(data_chunk.processed_object['x']['preds'])
                valid_data_labels.append(data_chunk.processed_object['label'])
                valid_data_real.append(data_chunk.processed_object['is_real'])
                valid_data_events.append(data_chunk.processed_object['event'])

        train_data_grus = np.concatenate(train_data_grus)
        train_data_preds = np.concatenate(train_data_preds)
        train_data_labels = np.concatenate(train_data_labels)
        train_data_real = np.concatenate(train_data_real)
        train_data_events = np.concatenate(train_data_events)

        valid_data_grus = np.concatenate(valid_data_grus)
        valid_data_preds = np.concatenate(valid_data_preds)
        valid_data_labels = np.concatenate(valid_data_labels)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_events = np.concatenate(valid_data_events)

        np.savez(
            f'{processed_data.output_name}_train',
            grus=train_data_grus,
            preds=train_data_preds,
            labels=train_data_labels,  # predicted right point and point was real
            is_real=train_data_real,
            events=train_data_events
        )
        np.savez(
            f'{processed_data.output_name}_valid',
            grus=valid_data_grus,
            preds=valid_data_preds,
            labels=valid_data_labels,
            is_real=valid_data_real,
            events=valid_data_events
        )
        LOGGER.info(f'Saved train hits to: {processed_data.output_name}_train.npz')
        LOGGER.info(f'Saved valid hits to: {processed_data.output_name}_valid.npz')
