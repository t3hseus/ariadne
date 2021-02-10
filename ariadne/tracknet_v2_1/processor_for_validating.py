import logging
from typing import List, Tuple, Optional, Iterable
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData
from ariadne.tracknet_v2.model import TrackNETv2
import gin
import pandas as pd
import numpy as np
import itertools
from ariadne.transformations import Compose, StandardScale, ToCylindrical, \
    ConstraintsNormalize, MinMaxScale, DropSpinningTracks, DropFakes, DropShort
from tqdm import tqdm
from copy import deepcopy
LOGGER = logging.getLogger('ariadne.prepare')
import torch
from ariadne.tracknet_v2.metrics import point_in_ellipse

@gin.configurable(denylist=['df_chunk_data'])
class ValidDataChunk(DataChunk):
    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)

class ProcessedValidData(ProcessedData):

    def __init__(self, processed_data: List[ProcessedDataChunk]):
        super().__init__(processed_data)
        self.processed_data = processed_data

class ProcessedValidDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: pd.DataFrame,
                 output_name: str,
                 id: int):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name
        self.id = id


@gin.configurable(denylist=['data_df'])
class Valid_Processor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 radial_stations_constraints,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float):
        super().__init__(
            processor_name='Valid_Processor',
            output_dir=output_dir,
            data_df=data_df)

        self.transformer = Compose([
            DropSpinningTracks(),
            DropShort(),
            ToCylindrical(),
            ConstraintsNormalize(
                use_global_constraints=False,
                constraints=radial_stations_constraints,
                columns=('r', 'phi', 'z')
            ),
        ])
        self.output_name = (self.output_dir + '/test_' + name_suffix)
        self.n_times_oversampling = n_times_oversampling
        self.valid_size = valid_size
        self.chunks = []

    def generate_chunks_iterable(self) -> Iterable[ValidDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> ValidDataChunk:
        processed = self.transformer(chunk_df)
        return ValidDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: ValidDataChunk,
                         idx: str) -> ProcessedValidDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/tracknet_%s_%d' % (idx, chunk_id))
        self.chunks.append(chunk_id)
        return ProcessedValidDataChunk(chunk_df, output_name, chunk_id)

    def cartesian(self, df1, df2):
        rows = itertools.product(df1.iterrows(), df2.iterrows())
        df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
        df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
        df = df[(df['track_left'] != df['track_right'])]
        df = pd.concat([df, df_fakes], axis=0)
        df = df.sample(frac=1).reset_index(drop=True)
        return df.reset_index(drop=True)

    def postprocess_chunks(self,
                           chunks: List[ProcessedValidDataChunk]) -> ProcessedValidData:
        for chunk in chunks:
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_moment = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            last_station = df[df['station'] > 1][['phi', 'z', 'track']]
            print(last_station)
            last_station = df[df['station'] > 1][['phi', 'z']].values
            print(last_station.shape)
            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_moment.append(data[['px','py','pz']].values[-1])
                chunk_data_real.append(1)
            print('=====> id', chunk.id)
            print('Multiplicity: ', len(chunk_data_x))
            first_station = df[df['station'] == 0][['r', 'phi', 'z', 'track']]
            first_station.columns = ['r_left', 'phi_left', 'z_left', 'track_left']

            second_station = df[df['station'] == 1][['r', 'phi', 'z', 'track']]
            second_station.columns = ['r_right', 'phi_right', 'z_right', 'track_right']

            fake_tracks = self.cartesian(first_station, second_station)
            fake_tracks = fake_tracks.sample(n=int(fake_tracks.shape[0] / 10), random_state=1)
            for i, row in fake_tracks.iterrows():
                temp_data = np.zeros((2, 3))
                temp_data[0, :] = row[['r_left', 'phi_left', 'z_left']].values
                temp_data[1, :] = row[['r_right', 'phi_right', 'z_right']].values
                chunk_data_x.append(temp_data)
                chunk_data_y.append(chunk_data_y[0])
                chunk_data_moment.append(chunk_data_moment[0])
                chunk_data_real.append(0)
                chunk_data_len.append(2)

            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data_moment = np.stack(chunk_data_moment, axis=0)
            chunk_data_real = np.stack(chunk_data_real, axis=0)
            chunk_data_len = np.stack(chunk_data_len, axis=0)
            chunk_data_event = np.ones(chunk_data_len.shape)
            chunk_data_event *= chunk.id

            chunk_data_event_last_station = np.ones(last_station.shape[0])
            chunk_data_event_last_station *= chunk.id

            chunk_data = {'x': chunk_data_x,
                          'y': chunk_data_y,
                          'moment': chunk_data_moment,
                          'is_real': chunk_data_real,
                          'last_station': last_station,
                          'last_station_event': chunk_data_event_last_station,
                          'event': chunk_data_event,
                          'len': chunk_data_len
                          }
            chunk.processed_object = chunk_data
        return ProcessedValidData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedValidData):

        valid_data_inputs = []
        valid_data_y = []
        valid_data_len = []
        valid_data_real = []
        valid_data_moment = []
        valid_data_event = []
        valid_data_last_station = []
        valid_data_last_station_event = []

        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            valid_data_inputs.append(data_chunk.processed_object['x'])
            valid_data_y.append(data_chunk.processed_object['y'])
            valid_data_real.append(data_chunk.processed_object['is_real'])
            valid_data_len.append(data_chunk.processed_object['len'])
            valid_data_moment.append(data_chunk.processed_object['moment'])
            valid_data_event.append(data_chunk.processed_object['event'])
            valid_data_last_station.append(data_chunk.processed_object['last_station'])
            valid_data_last_station_event.append(data_chunk.processed_object['last_station_event'])

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_y = np.concatenate(valid_data_y)
        valid_data_len = np.concatenate(valid_data_len)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_moment = np.concatenate(valid_data_moment)
        valid_data_event = np.concatenate(valid_data_event)
        valid_data_last_station = np.concatenate(valid_data_last_station)
        valid_data_last_station_event = np.concatenate(valid_data_last_station_event)

        np.savez(self.output_name,
                 x=valid_data_inputs, y=valid_data_y, len=valid_data_len,
                 moments=valid_data_moment, is_real=valid_data_real,
                 events=valid_data_event)
        np.savez(self.output_name+'_last_station', hits=valid_data_last_station, events=valid_data_last_station_event)
        print('Saved hits to: ', self.output_name + '.npz')
        print('Saved last station hits to: ', self.output_name + '_last_station.npz')
