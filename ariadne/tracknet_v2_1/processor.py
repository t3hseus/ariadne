import logging
from typing import List, Tuple, Optional, Iterable
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData
import os
import gin
import pandas as pd
import numpy as np
from tqdm import tqdm
from ariadne.transformations import Compose, StandardScale, ToCylindrical, \
    ConstraintsNormalize, MinMaxScale, DropSpinningTracks, DropFakes, DropShort
from ariadne.preprocessing import BaseTransformer
from ariadne.utils import brute_force_hits_two_first_stations
LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(denylist=['df_chunk_data'])
class TracknetDataChunk(DataChunk):

    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)

class ProcessedTracknetDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: Optional,
                 output_name: str,
                 id: int):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name
        self.id = id

class ProcessedTracknetData(ProcessedData):

    def __init__(self,
                 processed_data: List[ProcessedTracknetDataChunk],
                 output_name=''):
        super().__init__(processed_data)
        self.processed_data = processed_data
        self.output_name = output_name

@gin.configurable(denylist=['data_df'])
class TrackNetV21Processor(DataProcessor):
    """This processor prepares data for classifier and TrackNet.
       Only input data is saved, so it is needed to use TrackNetV2 and TrackNetClassifier simultaneously.
       To prepare data, cartesian product is used, and real tracks are marked as True, synthetic as False.
       Some additional data for analysis is saved too (momentum of particle, event)"""
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float,
                 transforms: List[BaseTransformer] = None
                 ):
        super().__init__(
            processor_name='TrackNet_Explicit_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)
        self.output_name = f'{self.output_dir}/{name_suffix}'
        self.n_times_oversampling = n_times_oversampling
        self.valid_size = valid_size
        self.chunks = []

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        processed = self.transformer(chunk_df)
        return TracknetDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        chunk_df = chunk.df_chunk_data
        if chunk_df.empty:
            return ProcessedTracknetDataChunk(None, '', -1)
        chunk_id = int(chunk_df.event.values[0])
        output_name = f'{self.output_dir}/tracknet_{idx.replace(".txt", "")}'
        self.chunks.append(chunk_id)
        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            if chunk.processed_object is None or len(chunk.processed_object) == 0:
                continue
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_momentum = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            last_station = df[df['station'] > 1][['phi', 'z']].values
            chunk_data_event_last_station = np.full(len(last_station), chunk.id)
            multiplicity = grouped_df.ngroups
            if multiplicity == 0:
                LOGGER.warning(f'Multiplicity in chunk #{chunk.id} is 0! This chunk is skipped')
                chunk.processed_object = None
                continue
            chunk_data_x, chunk_data_y, chunk_data_real = brute_force_hits_two_first_stations(df)
            chunk_data_len = np.full(len(chunk_data_x), 2)
            chunk_data_event = np.full(len(chunk_data_x), chunk.id)

            LOGGER.info(f'=====> id {chunk.id}')

            chunk_data = {'x': {'inputs': chunk_data_x, 'input_lengths': chunk_data_len},
                          'y': chunk_data_y,
                          'is_real': chunk_data_real,
                          'event': chunk_data_event,
                          'last_station': last_station,
                          'last_station_event': chunk_data_event_last_station,
                          'multiplicity': multiplicity}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks, chunks[0].output_name)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        train_data_inputs = []
        train_data_y = []
        train_data_len = []
        train_data_real = []
        train_data_event = []
        train_data_last_station = []
        train_data_last_station_event = []

        valid_data_inputs = []
        valid_data_y = []
        valid_data_len = []
        valid_data_real = []
        valid_data_event = []
        valid_data_last_station = []
        valid_data_last_station_event = []

        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.id in train_chunks:
                initial_len = len(data_chunk.processed_object['y'])
                multiplicity = data_chunk.processed_object['multiplicity']
                max_len = int(multiplicity + (initial_len / self.n_times_oversampling))
                train_data_inputs.append(data_chunk.processed_object['x']['inputs'][0:max_len])
                train_data_len.append(data_chunk.processed_object['x']['input_lengths'][0:max_len])
                train_data_y.append(data_chunk.processed_object['y'][0:max_len])
                train_data_real.append(data_chunk.processed_object['is_real'][0:max_len])
                train_data_event.append(data_chunk.processed_object['event'][0:max_len])
                train_data_last_station.append(data_chunk.processed_object['last_station'])
                train_data_last_station_event.append(data_chunk.processed_object['last_station_event'])
            else:
                valid_data_inputs.append(data_chunk.processed_object['x']['inputs'])
                valid_data_len.append(data_chunk.processed_object['x']['input_lengths'])
                valid_data_y.append(data_chunk.processed_object['y'])
                valid_data_real.append(data_chunk.processed_object['is_real'])
                valid_data_event.append(data_chunk.processed_object['event'])
                valid_data_last_station.append(data_chunk.processed_object['last_station'])
                valid_data_last_station_event.append(data_chunk.processed_object['last_station_event'])

        train_data_inputs = np.concatenate(train_data_inputs)
        train_data_y = np.concatenate(train_data_y)
        train_data_len = np.concatenate(train_data_len)
        train_data_real = np.concatenate(train_data_real)
        train_data_event = np.concatenate(train_data_event)

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_y = np.concatenate(valid_data_y)
        valid_data_len = np.concatenate(valid_data_len)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_event = np.concatenate(valid_data_event)

        valid_data_last_station = np.concatenate(valid_data_last_station)
        train_data_last_station = np.concatenate(train_data_last_station)
        valid_data_last_station_event = np.concatenate(valid_data_last_station_event)
        train_data_last_station_event = np.concatenate(train_data_last_station_event)
        np.savez(f'{processed_data.output_name}_train', inputs=train_data_inputs, input_lengths=train_data_len, y=train_data_y,
                   is_real=train_data_real, events=train_data_event)
        np.savez(f'{processed_data.output_name}_valid', inputs=valid_data_inputs, input_lengths=valid_data_len, y=valid_data_y,
                   is_real=valid_data_real, events=valid_data_event)
        np.savez(f'{processed_data.output_name}_train_last_station', hits=train_data_last_station, events=train_data_last_station_event)
        np.savez(f'{processed_data.output_name}_valid_last_station', hits=valid_data_last_station, events=valid_data_last_station_event)
        LOGGER.info(f'Saved last station hits to: {processed_data.output_name}_train_last_station.npz, {processed_data.output_name}_valid_last_station.npz')
