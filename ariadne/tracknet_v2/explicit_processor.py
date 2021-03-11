import logging
import itertools
import os
from typing import List, Tuple, Optional, Iterable

import gin
import pandas as pd
import numpy as np
from tqdm import tqdm

from ariadne.preprocessing import (
    BaseTransformer,
    DataProcessor,
    DataChunk,
    ProcessedDataChunk,
    ProcessedData
)
from ariadne.tracknet_v2.processor import (
    TracknetDataChunk,
    ProcessedTracknetData,
    TrackNetProcessor
)
from ariadne.utils import brute_force_hits_two_first_stations
LOGGER = logging.getLogger('ariadne.prepare')

class ProcessedTracknetDataChunk(ProcessedDataChunk):
    def __init__(self,
                 processed_object: Optional,
                 output_name: str,
                 event: int):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name
        self.id = event


@gin.configurable(denylist=['data_df'])
class TrackNetV2ExplicitProcessor(TrackNetProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 transforms: List[BaseTransformer] = None,
                 n_times_oversampling=10):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)
        self.output_name = f'{self.output_dir}/tracknet_{name_suffix}'
        self.n_times_oversampling = n_times_oversampling

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return ProcessedTracknetDataChunk(None, '', 0)

        chunk_id = int(chunk_df.event.values[0])
        output_name = f'{self.output_dir}/tracknet{idx}_{chunk_id}'
        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            if chunk.processed_object is None:
                continue
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_momentum = []
            chunk_data_event = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_momentum.append(data[['px', 'py', 'pz']].values[0])
                chunk_data_real.append(1)
                chunk_data_event.append(chunk.id)
            fake_tracks = brute_force_hits_two_first_stations(df)
            for i, row in tqdm(fake_tracks.iterrows()):
                temp_data = np.zeros((2, 3))
                temp_data[0, :] = row[['r_left', 'phi_left', 'z_left']].values
                temp_data[1, :] = row[['r_right', 'phi_right', 'z_right']].values
                chunk_data_x.append(temp_data)
                chunk_data_y.append(chunk_data_y[0])
                chunk_data_momentum.append(chunk_data_momentum[0])
                chunk_data_real.append(0)
                chunk_data_len.append(2)
                chunk_data_event.append(chunk.id)
            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data_momentum = np.stack(chunk_data_momentum, axis=0)
            chunk_data_real = np.stack(chunk_data_real, axis=0)
            chunk_data_event = np.stack(chunk_data_event, axis=0)
            chunk_data = {
                'x': {
                    'inputs': chunk_data_x,
                    'input_lengths': chunk_data_len},
                'y': chunk_data_y,
                'momentum': chunk_data_momentum,
                'is_real': chunk_data_real,
                'event': chunk_data_event
            }
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        all_data_inputs = []
        all_data_y = []
        all_data_len = []
        all_data_real = []
        all_data_momentum = []
        all_data_event = []
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            all_data_inputs.append(data_chunk.processed_object['x']['inputs'])
            all_data_len.append(data_chunk.processed_object['x']['input_lengths'])
            all_data_y.append(data_chunk.processed_object['y'])
            all_data_real.append(data_chunk.processed_object['is_real'])
            all_data_momentum.append(data_chunk.processed_object['momentum'])
            all_data_event.append(data_chunk.processed_object['event'])
        all_data_inputs = np.concatenate(all_data_inputs)
        all_data_y = np.concatenate(all_data_y)
        all_data_len = np.concatenate(all_data_len)
        all_data_real = np.concatenate(all_data_real)
        all_data_event = np.concatenate(all_data_event)
        all_data_momentum = np.concatenate(all_data_momentum)
        np.savez(
            self.output_name,
            inputs=all_data_inputs,
            input_lengths=all_data_len,
            y=all_data_y,
            momentums=all_data_momentum,
            is_real=all_data_real,
            events=all_data_event
        )
        LOGGER.info(f'Saved to: {self.output_name}.npz')
        np.savez(
            f'{self.output_name}_all_last_station',
            hits=all_data_y,
            event=all_data_event
        )
        LOGGER.info(f'Saved only last station hits to: {self.output_name}_last_station.npz')
