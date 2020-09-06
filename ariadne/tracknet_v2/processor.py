import logging
from typing import List, Tuple, Optional, Iterable
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData

import gin
import pandas as pd
import numpy as np

from transformations import Compose, StandartScale, ToCylindrical, \
    ConstraintsNormalize, MinMaxScale, DropSpinningTracks, DropFakes, DropShort

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
class TrackNet_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints,
                 df_suffixes: Tuple[str]
                 ):
        super().__init__(
            processor_name='TrackNet_v2_Processor',
            output_dir=output_dir,
            data_df=data_df)

        self._suffixes_df = df_suffixes
        self.transformer = Compose([
            DropFakes(),
            DropSpinningTracks(),
            DropShort(),
            ToCylindrical(),
            MinMaxScale(),
        ])

    def generate_chunks_iterable(self) -> Iterable[TracknetDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> TracknetDataChunk:
        processed = self.transformer(chunk_df)
        return TracknetDataChunk(processed)

    @gin.configurable(blacklist=['chunk', 'idx'])
    def preprocess_chunk(self,
                         chunk: TracknetDataChunk,
                         idx: int) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/tracknet_%d' % chunk_id)

        return ProcessedTracknetDataChunk(chunk_df, output_name)

    def postprocess_chunks(self,
                           chunks: List[ProcessedTracknetDataChunk]) -> ProcessedData:
        return ProcessedData(chunks)

    def save_on_disk(self,
                     processed_data: ProcessedTracknetData):
        for data_chunk in processed_data:
            if data_chunk.processed_object is None:
                continue
            filename = data_chunk.output_name
            data = data_chunk.processed_object
            np.savez(filename, data)