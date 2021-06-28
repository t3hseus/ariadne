import logging
import os
import sys

from typing import List, Tuple, Optional, Iterable

import gin
import pandas as pd
import numpy as np

from ariadne.preprocessing import (
    BaseTransformer,
    DataProcessor,
    DataChunk,
    ProcessedDataChunk,
    ProcessedData
)
from ariadne.tracknet_v2.processor import (TrackNetProcessor,
                                           TracknetDataChunk,
                                           ProcessedTracknetDataChunk,
                                           ProcessedTracknetData)
LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(denylist=['data_df'])
class TrackNetBMNProcessor(TrackNetProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 max_len=6,
                 det_indices=(0, 1),
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms,
            name_suffix='bmn')
        self.output_name = os.path.join(self.output_dir, f'tracknet_{name_suffix}')
        self.max_len = max_len
        self.det_indices = det_indices

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        if len(self.det_indices) > 1:
            chunk_df.loc[chunk_df.det == 1, 'station'] = chunk_df.loc[chunk_df.det == 1, 'station'].values + 3
        processed = self.transformer(chunk_df)
        return TracknetDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        chunk_df = chunk.df_chunk_data
        if chunk_df.empty:
            return ProcessedTracknetDataChunk(None, '')
        chunk_id = int(chunk_df.event.values[0])
        output_name = f'{self.output_dir}/tracknet_{idx.replace(".txt", "")}'
        return ProcessedTracknetDataChunk(chunk_df, output_name)


    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        for chunk in chunks:
            if chunk.processed_object is None:
                continue
            chunk_data_x = []
            chunk_data_y = []
            chunk_data_len = []
            df = chunk.processed_object
            grouped_df = df[df['track'] != -1].groupby('track')
            for i, data in grouped_df:
                if len(data) > self.max_len:
                    continue
                x_data = np.zeros((self.max_len, 3), dtype='float')
                x_values = data[['z', 'x', 'y']].values[:-1]
                x_data[:len(x_values)] = x_values
                chunk_data_x.append(x_data)
                chunk_data_y.append(data[['x', 'y']].values[-1])
                chunk_data_len.append(len(x_values))

            if len(chunk_data_x) == 0 or len(chunk_data_y) == 0:
                LOGGER.info(f'Chunk is empty! Skipping')
                chunk.processed_object = None
                continue
            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)

            chunk_data = {
                'x': {
                    'inputs': chunk_data_x,
                    'input_lengths': chunk_data_len},
                'y': chunk_data_y}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks[1].output_name, chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        all_data_inputs = []
        all_data_y = []
        all_data_len = []
        for data_chunk in processed_data.processed_data:
            if data_chunk.processed_object is None:
                continue
            all_data_inputs.append(data_chunk.processed_object['x']['inputs'])
            all_data_len.append(data_chunk.processed_object['x']['input_lengths'])
            all_data_y.append(data_chunk.processed_object['y'])
        all_data_inputs = np.concatenate(all_data_inputs).astype('float32')
        all_data_y = np.concatenate(all_data_y).astype('float32')
        all_data_len = np.concatenate(all_data_len)
        np.savez(
            processed_data.output_name,
            inputs=all_data_inputs,
            input_lengths=all_data_len, y=all_data_y
        )
        LOGGER.info(f'Saved to: {processed_data.output_name}.npz')
