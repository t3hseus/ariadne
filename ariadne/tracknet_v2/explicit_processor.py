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
                 output_name: str):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name


@gin.configurable(blacklist=['data_df'])
class TrackNet_Explicit_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 radial_stations_constraints,
                 name_suffix: str
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

        return ProcessedTracknetDataChunk(chunk_df, output_name)

    def cartesian(self, df1, df2):
        rows = itertools.product(df1.iterrows(), df2.iterrows())
        df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
        df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
        df = df[(df['track_left'] != df['track_right'])]
        df = pd.concat([df, df_fakes], axis=0)
        return df.reset_index(drop=True)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_moment = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['r', 'phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_moment.append(data[['px','py','pz']].values[-1])
                chunk_data_real.append(1)

            first_station = df[df['station'] == 0][['r', 'phi', 'z', 'track']]
            first_station.columns = ['r_left', 'phi_left', 'z_left', 'track_left']
            second_station = df[df['station'] == 1][['r', 'phi', 'z', 'track']]
            second_station.columns = ['r_right', 'phi_right', 'z_right', 'track_right']
            fake_tracks = self.cartesian(first_station, second_station)
            print(fake_tracks)
            print(chunk_data_x[-1].shape)
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
            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data_moment = np.stack(chunk_data_moment, axis=0)
            chunk_data_real = np.stack(chunk_data_real, axis=0)
            chunk_data = {'x': {'inputs': chunk_data_x, 'input_lengths': chunk_data_len},
                          'y': chunk_data_y, 'moment': chunk_data_moment, 'is_real': chunk_data_real}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        all_data_inputs = []
        all_data_y = []
        all_data_len = []
        all_data_real = []
        all_data_moment = []
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            all_data_inputs.append(data_chunk.processed_object['x']['inputs'])
            all_data_len.append(data_chunk.processed_object['x']['input_lengths'])
            all_data_y.append(data_chunk.processed_object['y'])
            all_data_real.append(data_chunk.processed_object['is_real'])
            all_data_moment.append(data_chunk.processed_object['moment'])
        all_data_inputs = np.concatenate(all_data_inputs)
        all_data_y = np.concatenate(all_data_y)
        all_data_len = np.concatenate(all_data_len)
        print(all_data_len.shape)
        all_data_real = np.concatenate(all_data_real)
        all_data_moment = np.concatenate(all_data_moment)
        np.savez(self.output_name, inputs=all_data_inputs, input_lengths=all_data_len, y=all_data_y,
                 moments=all_data_moment, is_real=all_data_real)
        print('Saved last station hits to: ', self.output_name + '.npz')
        np.savez(self.output_name+'_all_last_station', hits=np.unique(all_data_y, axis=0))
        print('Saved last station hits to: ', self.output_name+'_last_station.npz')
