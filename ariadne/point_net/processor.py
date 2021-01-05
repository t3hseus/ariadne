import logging
from typing import Tuple, Iterable, Optional, List

import gin
import pandas as pd
import numpy as np

from ariadne.point_net.point.points import Points, save_points_new
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData
from ariadne.transformations import Compose, ConstraintsNormalize, ToCylindrical

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(blacklist=['df_chunk_data'])
class PointsDataChunk(DataChunk):

    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)


class TransformedPointsDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: Optional[Points],
                 output_name: str):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name


class ProcessedPointsData(ProcessedData):

    def __init__(self, processed_data: List[TransformedPointsDataChunk]):
        super().__init__(processed_data)
        self.processed_data = processed_data


@gin.configurable(blacklist=['data_df'])
class PointNet_Processor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints
                 ):
        super().__init__(
            processor_name='PointNet_Processor',
            output_dir=output_dir,
            data_df=data_df)
        self.stations_constraints = stations_constraints
        self.transformer = None

    def generate_chunks_iterable(self) -> Iterable[PointsDataChunk]:
        self.transformer = Compose([
            ConstraintsNormalize(
                use_global_constraints=False,
                constraints=self.stations_constraints
            ),
            ToCylindrical(drop_old=True, cart_columns=('y', 'x'))
        ])
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> PointsDataChunk:
        processed = self.transformer(chunk_df)
        return PointsDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/points_%s_%d' % (idx, chunk_id))
        out = (chunk_df[['r', 'phi', 'z']].values / [1., np.pi, 1.]).T
        # if not chunk_df[(chunk_df.track != -1) & (chunk_df.track < 0) ].empty:
        #    logging.info("\nPointNet_Processor got hit with id < -1!\n for event %d" % chunk_id)
        #    return TransformedPointsDataChunk(None, "")
        out = Points(
            X=out.astype(np.float32),
            track=(chunk_df[['track']] >= 0).values.squeeze(-1).astype(np.float32)
        )
        return TransformedPointsDataChunk(out, output_name)

    def postprocess_chunks(self,
                           chunks: List[TransformedPointsDataChunk]) -> ProcessedPointsData:
        return ProcessedPointsData(chunks)

    def save_on_disk(self,
                     processed_data: ProcessedPointsData):
        save_points_new(processed_data.processed_data)


@gin.configurable(blacklist=['data_df'])
class PointNet_ProcessorBMN7(PointNet_Processor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints
                 ):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df, stations_constraints=stations_constraints)
        self.maxis = {'x':-1e7, 'y':-1e7, 'z':-1e7}
        self.minis = {'x': 1e7, 'y': 1e7, 'z': 1e7}
        self.mean_hits = []

    def generate_chunks_iterable(self) -> Iterable[PointsDataChunk]:
        self.transformer = Compose([
            ConstraintsNormalize(
                use_global_constraints=True,
                constraints=self.stations_constraints
            )
        ])
        return self.data_df.groupby('event')

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/points_%s_%d' % (idx, chunk_id))
        for col in ['x', 'y', 'z']:
            self.maxis[col] = max(chunk_df[col].max(), self.maxis[col])
            self.minis[col] = min(chunk_df[col].min(), self.maxis[col])

        out = chunk_df[['x', 'y', 'z']].values.T
        # if not chunk_df[(chunk_df.track != -1) & (chunk_df.track < 0) ].empty:
        #    logging.info("\nPointNet_Processor got hit with id < -1!\n for event %d" % chunk_id)
        #    return TransformedPointsDataChunk(None, "")
        out = Points(
            X=out.astype(np.float32),
            track=(chunk_df[['track']] >= 0).values.squeeze(-1).astype(np.float32)
        )
        self.mean_hits.append(len(chunk_df))
        return TransformedPointsDataChunk(out, output_name)

    def save_on_disk(self,
                     processed_data: ProcessedPointsData):
        print("\n\n=================\nSome stats:")
        for col in ['x', 'y', 'z']:
            print("MAXIS %s: %f" % (col, self.maxis[col]))
            print("MINIS %s: %f" % (col, self.minis[col]))
        print("Mean hits per event: %f" % (np.mean(self.mean_hits)))
        print("End\n===============\n")
        super(PointNet_ProcessorBMN7, self).save_on_disk(processed_data)