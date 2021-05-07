import logging
import os
from typing import List

import pandas as pd
import numpy as np
import torch
import gin
from ariadne.preprocessing import BaseTransformer
from ariadne.utils import brute_force_hits_two_first_stations, find_nearest_hit
from ariadne.tracknet_v2_1.processor import TrackNetV21Processor, ProcessedTracknetDataChunk, ProcessedTracknetData

LOGGER = logging.getLogger('ariadne.prepare-hydra-wombat')

@gin.configurable(denylist=['data_df'])
class ValidProcessor(TrackNetV21Processor):
    """This processor prepares data for validating of TrackNetClassifier and TrackNetV2.
       Only input data is saved, so it is needed to use TrackNetV2 and TrackNetClassifier simultaneously or use only TrackNetV2.
       To prepare-hydra-wombat data, cartesian product is used, and real tracks are marked as True, synthetic as False.
       Some additional data for analysis is saved too (momentum of particle, event).

       Validating needs to be done event-by-event, so to validate models, it is needed to group prepared data event-wise.

       Validating includes search of next hit of track, so last station data for each event is stored."""
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float,
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms,
            name_suffix=name_suffix,
            n_times_oversampling=n_times_oversampling,
            valid_size=valid_size)
        self.output_name = os.path.join(self.output_dir, f'test_{name_suffix}')
        self.n_times_oversampling = 1
        self.valid_size = valid_size
        self.chunks = []

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            if chunk.processed_object is None:
                continue
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            last_station = df[df['station'] > 1][['phi', 'z']].values
            chunk_data_event_last_station = np.full(len(last_station), chunk.id)
            multiplicity = grouped_df.ngroups
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            x, y, momentum, real = brute_force_hits_two_first_stations(df, return_momentum=True)
            chunk_data_len = np.full(len(x), 2)
            chunk_data_event = np.full(len(x), chunk.id)
            chunk_data = {'x': {'inputs': x, 'input_lengths': chunk_data_len},
                          'y': y,
                          'momentum': momentum,
                          'is_real': real,
                          'event': chunk_data_event,
                          'multiplicity': multiplicity,
                          'last_station': last_station,
                          'last_station_event': chunk_data_event_last_station}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks, chunks[0].output_name)

    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        valid_data_inputs = []
        valid_data_y = []
        valid_data_len = []
        valid_data_real = []
        valid_data_momentum = []
        valid_data_event = []
        valid_data_last_station = []
        valid_data_last_station_event = []

        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            valid_data_inputs.append(data_chunk.processed_object['x']['inputs'])
            valid_data_y.append(data_chunk.processed_object['y'])
            valid_data_real.append(data_chunk.processed_object['is_real'])
            valid_data_len.append(data_chunk.processed_object['x']['input_lengths'])
            valid_data_momentum.append(data_chunk.processed_object['momentum'])
            valid_data_event.append(data_chunk.processed_object['event'])
            valid_data_last_station.append(data_chunk.processed_object['last_station'])
            valid_data_last_station_event.append(data_chunk.processed_object['last_station_event'])

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_y = np.concatenate(valid_data_y)
        valid_data_len = np.concatenate(valid_data_len)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_momentum = np.concatenate(valid_data_momentum)
        valid_data_event = np.concatenate(valid_data_event)
        valid_data_last_station = np.concatenate(valid_data_last_station)
        valid_data_last_station_event = np.concatenate(valid_data_last_station_event)
        np.savez(
            processed_data.output_name,
            x=valid_data_inputs,
            y=valid_data_y,
            len=valid_data_len,
            momentums=valid_data_momentum,
            is_real=valid_data_real,
            events=valid_data_event
        )
        np.savez(
            f'{processed_data.output_name}_last_station',
            hits=valid_data_last_station,
            events=valid_data_last_station_event
        )
        LOGGER.info(f'Saved hits to: {processed_data.output_name}.npz')
        LOGGER.info(f'Saved last station hits to: {processed_data.output_name}_last_station.npz')
