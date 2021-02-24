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
from ariadne.utils import cartesian_product_two_stations, find_nearest_hit

LOGGER = logging.getLogger('ariadne.prepare')



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
class TrackNetV21ProcessorWithModel(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 n_times_oversampling: int,
                 valid_size: float,
                 device: str,
                 tracknet_v2_model: TrackNETv2,
                 tracknet_v2_checkpoint: str = '',
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            processor_name='TrackNet_v2_1_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)

        self.output_name = os.path.join(self.output_dir, f'tracknet_{name_suffix}')
        self.n_times_oversampling = n_times_oversampling
        self.valid_size = valid_size
        self.chunks = []
        self.device = torch.device(device)
        # TODO: load the whole model from checkpoint with config
        self.model = tracknet_v2_model

        if tracknet_v2_checkpoint and os.path.isfile(tracknet_v2_checkpoint):
            self.model = self.weights_update(
                model=self.model,
                checkpoint=tracknet_v2_checkpoint
            )

    def weights_update(self, model, checkpoint):
        # TODO: save torch model near the checkpoint
        pretrained_dict = torch.load(checkpoint).to(self.device)['state_dict']
        real_dict = {
            param: pretrained_dict[f'model.{param}']
                for param in model.state_dict()
        }
        model.load_state_dict(real_dict)
        model.eval()
        return model

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
            return ProcessedTracknetDataChunk(None, '', 0)

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'tracknet{idx}_{chunk_id}')
        self.chunks.append(chunk_id)
        return ProcessedTracknetDataChunk(chunk_df, output_name, chunk_id)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        # TODO: this code requires check (unittest)
        for chunk in chunks:
            if chunk.processed_object is None:
                continue
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            chunk_data_real = []
            chunk_data_momentum = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')

            for i, data in grouped_df:
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['phi', 'z']].values[-1])
                chunk_data_len.append(2)
                chunk_data_momentum.append(data[['px','py','pz']].values[-1])
                chunk_data_real.append(1)
            LOGGER.info(f'=====> id {chunk.id}')
            fake_tracks = cartesian_product_two_stations(df)
            for i, row in tqdm(fake_tracks.iterrows()):
                temp_data = np.zeros((2, 3))
                temp_data[0, :] = row[['r_left', 'phi_left', 'z_left']].values
                temp_data[1, :] = row[['r_right', 'phi_right', 'z_right']].values
                chunk_data_x.append(temp_data)
                chunk_data_y.append(chunk_data_y[0])
                chunk_data_momentum.append(chunk_data_momentum[0])
                chunk_data_real.append(0)
                chunk_data_len.append(2)

            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data_momentum = np.stack(chunk_data_momentum, axis=0)
            chunk_data_real = np.stack(chunk_data_real, axis=0)
            chunk_data_len = np.stack(chunk_data_len, axis=0)

            chunk_data_event = np.full(len(chunk_data_x), chunk.id)
            #chunk_data_event_last_station = np.full(len(last_station), chunk.id)

            last_station_hits = torch.from_numpy(chunk_data_y[:, 1:]).to(self.device)
            chunk_prediction = self.model(torch.tensor(chunk_data_x).to(self.device),
                                           torch.tensor(chunk_data_len, dtype=torch.int64).to(self.device))
            chunk_gru = self.model.last_gru_output.detach().cpu().numpy()
            nearest_hits, in_ellipse = find_nearest_hit(chunk_prediction, last_station_hits)
            is_prediction_true = torch.isclose(last_station_hits.to(torch.float32).to(self.device), nearest_hits.to(torch.float32).to(self.device))
            is_prediction_true = is_prediction_true.sum(dim=1) / 2.
            found_right_points = (in_ellipse).detach().cpu().numpy() & (chunk_data_real == 1) & (is_prediction_true.detach().cpu().numpy() == 1)
            chunk_data = {'x': {'gru': chunk_gru, 'preds': nearest_hits},
                          'y': found_right_points,
                          'momentum': chunk_data_momentum,
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
        train_data_momentum = []
        train_data_event = []

        valid_data_inputs = []
        valid_data_y = []
        valid_data_len = []
        valid_data_real = []
        valid_data_momentum = []
        valid_data_event = []

        train_chunks = np.random.choice(self.chunks, int(len(self.chunks) * (1-self.valid_size)), replace=False)
        #print(train_chunks)
        valid_chunks = list(set(self.chunks) - set(train_chunks))
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            if data_chunk.id in train_chunks:
                y = data_chunk.processed_object['y']
                max_len = int(0.5*len(y))
                train_data_inputs.append(data_chunk.processed_object['x']['gru'][0:max_len])
                train_data_len.append(data_chunk.processed_object['x']['preds'][0:max_len].detach().cpu().numpy())
                train_data_y.append(data_chunk.processed_object['y'][0:max_len])
                train_data_real.append(data_chunk.processed_object['is_real'][0:max_len])
                train_data_momentum.append(data_chunk.processed_object['momentum'][0:max_len])
                train_data_event.append(data_chunk.processed_object['event'][0:max_len])
            else:
                valid_data_inputs.append(data_chunk.processed_object['x']['gru'])
                valid_data_len.append(data_chunk.processed_object['x']['preds'].detach().cpu().numpy())
                valid_data_y.append(data_chunk.processed_object['y'])
                valid_data_real.append(data_chunk.processed_object['is_real'])
                valid_data_momentum.append(data_chunk.processed_object['momentum'])
                valid_data_event.append(data_chunk.processed_object['event'])

        train_data_inputs = np.concatenate(train_data_inputs)
        train_data_y = np.concatenate(train_data_y)
        train_data_len = np.concatenate(train_data_len)
        train_data_real = np.concatenate(train_data_real)
        train_data_momentum = np.concatenate(train_data_momentum)
        train_data_event = np.concatenate(train_data_event)

        valid_data_inputs = np.concatenate(valid_data_inputs)
        valid_data_y = np.concatenate(valid_data_y)
        valid_data_len = np.concatenate(valid_data_len)
        valid_data_real = np.concatenate(valid_data_real)
        valid_data_momentum = np.concatenate(valid_data_momentum)
        valid_data_event = np.concatenate(valid_data_event)

        np.savez(
            f'{self.output_name}_train',
            gru=train_data_inputs,
            preds=train_data_len,
            y=train_data_y,
            momentums=train_data_momentum,
            is_real=train_data_real,
            events=train_data_event
        )
        np.savez(
            f'{self.output_name}_valid',
            gru=valid_data_inputs,
            preds=valid_data_len,
            y=valid_data_y,
            momentums=valid_data_momentum,
            is_real=valid_data_real,
            events=valid_data_event
        )
        LOGGER.info(f'Saved train hits to: {self.output_name}_train.npz')
        LOGGER.info(f'Saved valid hits to: {self.output_name}_valid.npz')
