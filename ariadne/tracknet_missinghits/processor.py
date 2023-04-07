import logging
import os
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

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(denylist=['df_chunk_data'])
class TracknetDataChunk(DataChunk):
    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)

class ProcessedTracknetData(ProcessedData):
    def __init__(self,
                 output_name: str,
                 processed_data: List[ProcessedDataChunk]
                 ):
        super().__init__(processed_data)
        self.processed_data = processed_data
        self.output_name = output_name

class ProcessedTracknetDataChunk(ProcessedDataChunk):
    def __init__(self,
                 processed_object: Optional,
                 output_name: str):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name

@gin.configurable(denylist=['data_df'])
class TrackNetProcessor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            processor_name='TrackNet_v2_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)
        self.output_name = os.path.join(self.output_dir, f'tracknet_{name_suffix}')

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
                chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
                chunk_data_y.append(data[['phi', 'z']].values[-1])
                chunk_data_len.append(2)
            chunk_data_x = np.stack(chunk_data_x, axis=0)
            chunk_data_y = np.stack(chunk_data_y, axis=0)
            chunk_data = {
                'x': {
                    'inputs': chunk_data_x,
                    'input_lengths': chunk_data_len},
                'y': chunk_data_y}
            chunk.processed_object = chunk_data
        return ProcessedTracknetData(chunks[0].output_name,chunks)

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

@gin.configurable(denylist=['data_df'])
class TrackNetProcessorWithMaskMissingHits(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 transforms: List[BaseTransformer] = None,
                 columns=('x', 'y', 'z'),
                 det_indices=(0,1),
                 min_track_len=4,
                 balance=True,
                 filter_first_n=0):
        super().__init__(
            processor_name='TrackNet_v2_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)
        self.output_name = os.path.join(self.output_dir, f'masked_tracknet_{name_suffix}')
        self.columns = columns
        self.det_indices = det_indices
        self.filter_first_stations = filter_first_n
        self.min_track_len=min_track_len - 1
        self.balance=balance

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        if len(self.det_indices) > 1:
            #self.data_df.loc[self.data_df.det == 1, 'station'] = self.data_df.loc[self.data_df.det == 1, 'station'].values + 3
            if self.filter_first_stations > 0:
                self.data_df = self.data_df.loc[self.data_df['station'] >= self.filter_first_stations, :]
                self.data_df.loc[:, 'station'] = self.data_df.loc[:, 'station'].values - self.filter_first_stations
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        processed = self.transformer(chunk_df)
        return TracknetDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        df = chunk.df_chunk_data
        if df.empty:
            return ProcessedTracknetDataChunk(None, '')
        chunk_data_xs = {}
        stations = df.groupby('station').size().max()
        # max_station = max(stations)
        if stations > 1000:
            return ProcessedTracknetDataChunk(None, '')
        
        grouped_df = df[df['track'] != -1].groupby('track')
        for i, data in grouped_df:
            track = data[list(self.columns)].values
            track_len = len(track)

            if track_len > self.min_track_len:
                if track_len not in chunk_data_xs.keys():
                    chunk_data_xs[track_len] = []
                chunk_data_xs[track_len].append(track)
        if len(chunk_data_xs.keys()) == 0:
            return ProcessedTracknetDataChunk(None, '')
        if self.balance:
            nums = [len(i) for i in chunk_data_xs.items()]
            min_num = min(nums)
            chunk_data_xs = {k: v[:min_num] for k,v in chunk_data_xs.items()}
        return ProcessedTracknetDataChunk(chunk_data_xs, self.output_name)


    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        return ProcessedTracknetData(self.output_name, chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        chunk_data_xs = []
        for chunk in processed_data.processed_data:
            if chunk.processed_object is None:
                continue
            for items in chunk.processed_object.values():
                chunk_data_xs.extend(items)
        all_data_inputs = np.array(chunk_data_xs, dtype=object)
        LOGGER.info(f'Get {len(all_data_inputs)} tracks')
        try:
            temp_inputs = np.load(self.output_name+'.npy', allow_pickle=True)
            all_data_inputs = np.concatenate((temp_inputs, all_data_inputs))
        except:
            LOGGER.info('new array is created')
        np.save(self.output_name, all_data_inputs, allow_pickle=True)
        temp_inputs = np.load(self.output_name+'.npy', allow_pickle=True)
        LOGGER.info(f'now have {len(temp_inputs)} tracks')
        LOGGER.info(f'Saved to: {self.output_name}.npy as object-pickle')

        
@gin.configurable(denylist=['data_df'])
class TrackNetProcessorWithMaskMissingHitsByStation(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 name_suffix: str,
                 transforms: List[BaseTransformer] = None,
                 columns=('x', 'y', 'z'),
                 det_indices=(0,1),
                 min_track_len=4,
                 balance=True,
                 filter_first_n=0):
        super().__init__(
            processor_name='TrackNet_v2_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms)
        self.output_name = os.path.join(self.output_dir, f'masked_tracknet_{name_suffix}')
        self.columns = columns
        self.det_indices = det_indices
        self.filter_first_stations = filter_first_n
        self.min_track_len=min_track_len - 1
        self.balance=balance

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        if len(self.det_indices) > 1:
            #self.data_df.loc[self.data_df.det == 1, 'station'] = self.data_df.loc[self.data_df.det == 1, 'station'].values + 3
            if self.filter_first_stations > 0:
                self.data_df = self.data_df.loc[self.data_df['station'] >= self.filter_first_stations, :]
                self.data_df.loc[:, 'station'] = self.data_df.loc[:, 'station'].values - self.filter_first_stations
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        processed = self.transformer(chunk_df)
        return TracknetDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: str) -> ProcessedTracknetDataChunk:
        df = chunk.df_chunk_data
        if df.empty:
            return ProcessedTracknetDataChunk(None, '')
        chunk_data_xs = {}
        stations = df.groupby('station').size().max()
        # max_station = max(stations)
        if stations > 1000:
            return ProcessedTracknetDataChunk(None, '')
        
        grouped_df = df[df['track'] != -1].groupby('track')
        for i, data in grouped_df:
            if len(data) == data['station'].max() + 1:
                track = data[list(self.columns)].values
                track_len = len(track)
            else:
                track_len = data['station'].max() + 1
                track = np.zeros((track_len, 3))
                track[data['station']] = data[list(self.columns)].values
                track = track[:track_len]
            
            if track_len > self.min_track_len:
                if track_len not in chunk_data_xs.keys():
                    chunk_data_xs[track_len] = []
                chunk_data_xs[track_len].append(track)
        if len(chunk_data_xs.keys()) == 0:
            return ProcessedTracknetDataChunk(None, '')
        if self.balance:
            nums = [len(i) for i in chunk_data_xs.items()]
            min_num = min(nums)
            chunk_data_xs = {k: v[:min_num] for k,v in chunk_data_xs.items()}
        return ProcessedTracknetDataChunk(chunk_data_xs, self.output_name)


    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedTracknetData:
        return ProcessedTracknetData(self.output_name, chunks)


    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        chunk_data_xs = []
        for chunk in processed_data.processed_data:
            if chunk.processed_object is None:
                continue
            for items in chunk.processed_object.values():
                chunk_data_xs.extend(items)
        all_data_inputs = np.array(chunk_data_xs, dtype=object)
        LOGGER.info(f'Get {len(all_data_inputs)} tracks')
        try:
            temp_inputs = np.load(self.output_name+'.npy', allow_pickle=True)
            all_data_inputs = np.concatenate((temp_inputs, all_data_inputs))
        except:
            LOGGER.info('new array is created')
        np.save(self.output_name, all_data_inputs, allow_pickle=True)
        temp_inputs = np.load(self.output_name+'.npy', allow_pickle=True)
        LOGGER.info(f'now have {len(temp_inputs)} tracks')
        LOGGER.info(f'Saved to: {self.output_name}.npy as object-pickle')
