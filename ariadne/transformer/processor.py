import logging
import os
from typing import Iterable, List

import gin
import numpy as np
import pandas as pd

from ariadne.point_net.point.points import Points, save_points_new
from ariadne.point_net.processor import (
    PointsDataChunk,
    ProcessedPointsData,
    TransformedPointsDataChunk,
)
from ariadne.preprocessing import BaseTransformer, DataProcessor, ProcessedDataChunk

LOGGER = logging.getLogger("ariadne.prepare")


@gin.configurable(denylist=["data_df"])
class CloudProcessor(DataProcessor):
    def __init__(
        self,
        output_dir: str,
        data_df: pd.DataFrame,
        n_events_per_chunk: int = 1,
        transforms: List[BaseTransformer] = None,
    ):
        super().__init__(
            processor_name="CloudProcessor",
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms,
        )
        self.n_events_per_chunk = n_events_per_chunk

    def generate_chunks_iterable(self) -> Iterable[PointsDataChunk]:
        self.data_df["event_group"] = self.data_df["event"] // self.n_events_per_chunk
        if self.n_events_per_chunk > 1:
            return self.data_df.groupby(["event_group"])
        else:
            return self.data_df.groupby(["event"])

    def construct_chunk(self, chunk_df: pd.DataFrame) -> PointsDataChunk:
        processed = self.transformer(chunk_df)
        return PointsDataChunk(processed)

    def preprocess_chunk(self, chunk: PointsDataChunk, idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f"points_{idx}_{chunk_id}")
        # out = (chunk_df[['r', 'phi', 'z']].values / [1., np.pi, 1.]).T
        out = chunk_df[["x", "y", "z"]].values.T
        # out = Points(
        #     X=out.astype(np.float32), track=chunk_df["track"].values.astype(np.float32)
        # )
        out = Points(
            X=out.astype(np.float32), n_tracks=len(np.unique(chunk_df["track"].values.astype(np.float32)))
        )
        return TransformedPointsDataChunk(out, output_name)

    def postprocess_chunks(
        self, chunks: List[TransformedPointsDataChunk]
    ) -> ProcessedPointsData:
        return ProcessedPointsData(chunks)

    def save_on_disk(self, processed_data: ProcessedPointsData):
        broken = 0
        total = len(processed_data.processed_data)
        for obj in processed_data.processed_data:
            if obj.processed_object is None:
                broken += 1

        LOGGER.info(f"\n==Collected {broken} broken events out of {total} events.==\n")
        save_points_new(processed_data.processed_data)
