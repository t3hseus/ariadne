import logging
import os
from typing import List

import gin
import pandas as pd
import numpy as np

from ariadne.transformations import BaseTransformer
from ariadne.tracknet_v2_1.processor import (
    ProcessedTracknetData,
    TrackNetV21Processor
)

LOGGER = logging.getLogger('ariadne.prepare')

@gin.configurable(denylist=['data_df'])
class ValidProcessor(TrackNetV21Processor):
    """This processor prepares data for validating of Classifier and TrackNetV2.
       Only input data is saved, so it is needed to use TrackNetV2 and Classifier simultaneously or use only TrackNetV2.
       To prepare data, cartesian product is used, and real tracks are marked as True, synthetic as False.
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
