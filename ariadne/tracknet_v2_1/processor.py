import logging
from typing import List, Tuple, Optional, Iterable
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData

import gin
import pandas as pd
import numpy as np
import itertools
from ariadne.transformations import Compose, StandardScale, ToCylindrical, \
    ConstraintsNormalize, MinMaxScale, DropSpinningTracks, DropFakes, DropShort
from tqdm import tqdm
LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(blacklist=['df_chunk_data'])
class TracknetDataChunk(DataChunk):

    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)

class ProcessedTracknetData(ProcessedData):

    def __init__(self, processed_data: List[ProcessedDataChunk]):
        super().__init__(processed_data)
        self.processed_data = processed_data

class ProcessedTracknetDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: Optional,
                 output_name: str,
                 id: int):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name
        self.id = id


@gin.configurable(blacklist=['data_df'])
class TrackNetV2_1_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 radial_stations_constraints,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float
                 ):
        super().__init__(
            processor_name='TrackNet_Explicit_Processor',
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
        self.output_name = (self.output_dir + '/tracknet_' + name_suffix)
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
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/tracknet_%s_%d' % (idx, chunk_id))
        self.chunks.append(chunk_id)

        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def cartesian(self, df1, df2):
        rows = itertools.product(df1.iterrows(), df2.iterrows())
        df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
        df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
        df = df[(df['track_left'] != df['track_right'])]
        df = pd.concat([df, df_fakes], axis=0)
        df = df.sample(frac=1).reset_index(drop=True)
        return df.reset_index(drop=True)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_moment = []
            chunk_data_event = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['r', 'phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_moment.append(data[['px','py','pz']].values[-1])
                chunk_data_real.append(1)
                chunk_data_event.append(chunk.id)
            print('num_real before oversampling:', len(chunk_data_x))
            chunk_data_x = chunk_data_x * self.n_times_oversampling
            chunk_data_y = chunk_data_y * self.n_times_oversampling
            chunk_data_len = chunk_data_len * self.n_times_oversampling
            chunk_data_moment = chunk_data_moment * self.n_times_oversampling
            chunk_data_real = chunk_data_real * self.n_times_oversampling
            print('num_real after oversampling:', len(chunk_data_x))

            first_station = df[df['station'] == 0][['r', 'phi', 'z', 'track']]
            first_station.columns = ['r_left', 'phi_left', 'z_left', 'track_left']

            second_station = df[df['station'] == 1][['r', 'phi', 'z', 'track']]
            second_station.columns = ['r_right', 'phi_right', 'z_right', 'track_right']

            fake_tracks = self.cartesian(first_station, second_station)

            for i, row in tqdm(fake_tracks.iterrows()):
                temp_data = np.zeros((2, 3))
                temp_data[0, :] = row[['r_left', 'phi_left', 'z_left']].values
                temp_data[1, :] = row[['r_right', 'phi_right', 'z_right']].values
                chunk_data_x.append(temp_data)
                #print(chunk_data_x[-1].shape)
                chunk_data_y.append(chunk_data_y[0])
                chunk_data_moment.append(chunk_data_moment[0])
                chunk_data_real.append(0)
                chunk_data_len.append(2)
                chunk_data_event.append(chunk.id)

            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data_moment = np.stack(chunk_data_moment, axis=0)
            chunk_data_real = np.stack(chunk_data_real, axis=0)
            chunk_data_event = np.stack(chunk_data_event, axis=0)
            chunk_data = {'x': {'inputs': chunk_data_x, 'input_lengths': chunk_data_len},
                          'y': chunk_data_y,
                          'moment': chunk_data_moment,
                          'is_real': chunk_data_real,
                          'event': chunk_data_event}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        train_data_inputs = []
        train_data_y = []
        train_data_len = []
        train_data_real = []
        train_data_moment = []
        train_data_event = []

        valid_data_inputs = []
        valid_data_y = []
        valid_data_len = []
        valid_data_real = []
        valid_data_moment = []
        valid_data_event = []
        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        valid_chunks = list(set(self.chunks) - set(train_chunks))
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.processed_object['event'] in train_chunks:
                train_data_inputs.append(data_chunk.processed_object['x']['inputs'])
                train_data_len.append(data_chunk.processed_object['x']['input_lengths'])
                train_data_y.append(data_chunk.processed_object['y'])
                train_data_real.append(data_chunk.processed_object['is_real'])
                train_data_moment.append(data_chunk.processed_object['moment'])
                train_data_event.append(data_chunk.processed_object['event'])
            else:
                valid_data_inputs.append(data_chunk.processed_object['x']['inputs'])
                valid_data_len.append(data_chunk.processed_object['x']['input_lengths'])
                valid_data_y.append(data_chunk.processed_object['y'])
                valid_data_real.append(data_chunk.processed_object['is_real'])
                valid_data_moment.append(data_chunk.processed_object['moment'])
                valid_data_event.append(data_chunk.processed_object['event'])

        train_data_inputs = np.concatenate(train_data_inputs)
        train_data_y = np.concatenate(train_data_y)
        train_data_len = np.concatenate(train_data_len)
        train_data_real = np.concatenate(train_data_real)
        train_data_moment = np.concatenate(train_data_moment)
        train_data_event = np.concatenate(train_data_event)

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_y = np.concatenate(valid_data_y)
        valid_data_len = np.concatenate(valid_data_len)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_moment = np.concatenate(valid_data_moment)
        valid_data_event = np.concatenate(valid_data_event)

        np.savez(self.output_name+'_train', inputs=train_data_inputs, input_lengths=train_data_len, y=train_data_y,
                 moments=train_data_moment, is_real=train_data_real, events=train_data_event)
        np.savez(self.output_name + '_valid', inputs=valid_data_inputs, input_lengths=valid_data_len, y=valid_data_y,
                 moments=valid_data_moment, is_real=valid_data_real, events=valid_data_event)
        print('Saved last station hits to: ', self.output_name + '.npz')
        np.savez(self.output_name+'_train_last_station', hits=np.unique(train_data_y, axis=0), events=train_data_event)
        np.savez(self.output_name + '_valid_last_station', hits=np.unique(valid_data_y, axis=0), events=valid_data_event)
        print('Saved last station hits to: ', self.output_name+'_train_last_station.npz', self.output_name+'_valid_last_station.npz')
