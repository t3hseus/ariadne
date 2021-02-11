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


@gin.configurable(denylist=['data_df'])
class TrackNetV2_1_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 radial_stations_constraints,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float,
                 path_to_checkpoint: str,
                 device: str):
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
        self.device = torch.device(device)
        self.model = self.weights_update(model=TrackNETv2(input_features=3, conv_features=32, rnn_type='gru', batch_first=True),
                                         checkpoint=torch.load(path_to_checkpoint)).to(self.device)

    def weights_update(self, model, checkpoint):
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        real_dict = {}
        for (k, v) in model_dict.items():
            needed_key = None
            for pretr_key in pretrained_dict:
                if k in pretr_key:
                    needed_key = pretr_key
                    break
            assert needed_key is not None, "key %s not in pretrained_dict %r!" % (k, pretrained_dict.keys())
            real_dict[k] = pretrained_dict[needed_key]

        model.load_state_dict(real_dict)
        model.eval()
        return model

    def find_nearest_hit(self, ellipses, y):
        centers = ellipses[:, :2]
        #print(centers)
        last_station_hits = deepcopy(y)
        #print(last_station_hits.size())
        # print('center', centers)
        # print(torch.isclose(torch.tensor(y, dtype=torch.double), last_station_hits))
        # print(self.last_station_hits[0:10])
        dists = torch.cdist(last_station_hits.float(), centers.float())
        #print(dists)
        minimal = last_station_hits[torch.argmin(dists, dim=0)]
        #print(minimal)
        is_in_ellipse = point_in_ellipse(ellipses, minimal)
        return minimal, is_in_ellipse

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
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['r', 'phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_moment.append(data[['px','py','pz']].values[-1])
                chunk_data_real.append(1)
            print('=====> id', chunk.id)
            first_station = df[df['station'] == 0][['r', 'phi', 'z', 'track']]
            first_station.columns = ['r_left', 'phi_left', 'z_left', 'track_left']

            second_station = df[df['station'] == 1][['r', 'phi', 'z', 'track']]
            second_station.columns = ['r_right', 'phi_right', 'z_right', 'track_right']

            fake_tracks = self.cartesian(first_station, second_station)
            fake_tracks = fake_tracks.sample(n=int(fake_tracks.shape[0] / 10), random_state=1)
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
            chunk_data_len = np.stack(chunk_data_len, axis=0)
            chunk_data_event = np.ones(chunk_data_len.shape)

            chunk_data_event *= chunk.id
            #print(chunk_data_event)
            last_station_hits = torch.from_numpy(chunk_data_y[:, 1:]).to(self.device)
            chunk_prediction = self.model(torch.tensor(chunk_data_x).to(self.device),
                                           torch.tensor(chunk_data_len, dtype=torch.int64).to(self.device))
            chunk_gru = self.model.last_gru_output.detach().cpu().numpy()
            nearest_hits, in_ellipse = self.find_nearest_hit(chunk_prediction, last_station_hits)
            is_prediction_true = torch.isclose(last_station_hits.to(torch.float32).to(self.device), nearest_hits.to(torch.float32).to(self.device))
            is_prediction_true = is_prediction_true.sum(dim=1) / 2.
            # second_is_prediction_true = torch.isclose(batch_target.to(torch.float32), second_nearest_points.to(torch.float32))
            # second_is_prediction_true = second_is_prediction_true.sum(axis=1) / 2.
            found_right_points = (in_ellipse).detach().cpu().numpy() & (chunk_data_real == 1) & (is_prediction_true.detach().cpu().numpy() == 1)
            chunk_data = {'x': {'gru': chunk_gru, 'preds': nearest_hits},
                          'y': found_right_points,
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
        #print(train_chunks)
        valid_chunks = list(set(self.chunks) - set(train_chunks))
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.id in train_chunks:
                #print(f"{data_chunk.processed_object['event'][0]} is in train data")

                y = data_chunk.processed_object['y']
                print(y)
                max_len = int(0.5*len(y))
                print(y[0:max_len])
                train_data_inputs.append(data_chunk.processed_object['x']['gru'][0:max_len])
                #print(train_data_inputs)
                train_data_len.append(data_chunk.processed_object['x']['preds'][0:max_len].detach().cpu().numpy())
                train_data_y.append(data_chunk.processed_object['y'][0:max_len])
                train_data_real.append(data_chunk.processed_object['is_real'][0:max_len])
                train_data_moment.append(data_chunk.processed_object['moment'][0:max_len])
                train_data_event.append(data_chunk.processed_object['event'][0:max_len])
            else:
                #print(f"{data_chunk.processed_object['event'][0]} is in valid data")
                valid_data_inputs.append(data_chunk.processed_object['x']['gru'])
                valid_data_len.append(data_chunk.processed_object['x']['preds'].detach().cpu().numpy())
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

        np.savez(self.output_name+'_train', gru=train_data_inputs, preds=train_data_len, y=train_data_y,
                 moments=train_data_moment, is_real=train_data_real, events=train_data_event)
        np.savez(self.output_name + '_valid', gru=valid_data_inputs, preds=valid_data_len, y=valid_data_y,
                 moments=valid_data_moment, is_real=valid_data_real, events=valid_data_event)
        print('Saved last station hits to: ', self.output_name + '_train/_valid.npz')
