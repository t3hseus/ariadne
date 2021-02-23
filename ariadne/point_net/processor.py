import logging
import os
from typing import Tuple, Iterable, Optional, List

import gin
import pandas as pd
import numpy as np

from ariadne.point_net.point.points import Points, save_points_new
from ariadne.preprocessing import (
    BaseTransformer,
    DataProcessor,
    DataChunk,
    ProcessedDataChunk,
    ProcessedData
)

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(denylist=['df_chunk_data'])
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


@gin.configurable()
class PointNet_Processor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 transforms: List[BaseTransformer] = None):
        super().__init__(processor_name='PointNet_Processor', output_dir=output_dir, transforms=transforms)

    def generate_chunks_iterable(self, data_df) -> Iterable[PointsDataChunk]:
        return data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> PointsDataChunk:
        processed = self.transformer(chunk_df)
        return PointsDataChunk(processed)

    def total_chunks(self, data_df) -> int:
        return data_df['event'].nunique()

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'points_{idx}_{chunk_id}')
        out = (chunk_df[['r', 'phi', 'z']].values / [1., np.pi, 1.]).T
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
        broken = 0
        total = len(processed_data.processed_data)
        for obj in processed_data.processed_data:
            if obj.processed_object is None:
                broken += 1

        LOGGER.info(f'\n==Collected {broken} broken events out of {total} events.==\n')
        save_points_new(processed_data.processed_data)


@gin.configurable
class PointNet_ProcessorBMN7(PointNet_Processor):
    def __init__(self,
                 output_dir: str,
                 coord_cols,
                 stats_cols,
                 det_indices,
                 forward_fields,
                 with_random_sort,
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            output_dir=output_dir,
            transforms=transforms
        )
        self.coord_cols = coord_cols
        self.stats_cols = stats_cols
        self.det_indices = det_indices
        self._forward_fields = forward_fields
        self.maxis = {col: -1e7 for col in stats_cols}
        self.minis = {col: 1e7 for col in stats_cols}
        self.mean_hits = []
        self.mean_tracks = []
        self.mean_len_tracks = []
        self.with_random_sort = with_random_sort

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'points_{idx}_{chunk_id}')

        chunk_df = chunk_df[chunk_df.det.isin(self.det_indices)]
        if len(self.det_indices) > 1:
            chunk_df.loc[chunk_df.det == 1, 'station'] = chunk_df.loc[chunk_df.det == 1, 'station'].values + 3

        if chunk_df.empty:
            LOGGER.warning(f'\nPointNet_Processor empty event  {chunk_id}')
            return TransformedPointsDataChunk(None, '')

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.warning(f'\nPointNet_Processor got hit with id < -1!\n for event {chunk_id}')
            return TransformedPointsDataChunk(None, '')

        tracks = chunk_df[chunk_df.track != -1]

        mean_lens = []
        for tr_id, track in tracks.groupby('track'):
            while not track[track.station.duplicated()].empty:
                copyied = track.copy()
                copyied.loc[track.station.duplicated().index[:-1], 'track'] = -1
                chunk_df.loc[chunk_df.track == tr_id] = copyied
                track = chunk_df[chunk_df.track == tr_id]

            if len(track) < 5:
                chunk_df.loc[chunk_df.track == tr_id, 'track'] = -1
                continue

            mean_lens.append(len(track))


        if len(mean_lens) > 0:
            self.mean_tracks.append(chunk_df[chunk_df.track != -1].track.nunique())
            self.mean_len_tracks.append(np.mean(mean_lens))
        # if chunk_df[chunk_df.track == -1].empty:
        #    LOGGER.info(f'\nPointNet_Processor got event with no fakes!\n for event {chunk_id}')
        #    return TransformedPointsDataChunk(None, '')

        # if chunk_df[chunk_df.track != -1].empty:
        #    LOGGER.info(f'\nPointNet_Processor got event with no real hits!\n for event {chunk_id}')
        #    return TransformedPointsDataChunk(None, '')

        for col in self.stats_cols:
            self.maxis[col] = max(chunk_df[col].max(), self.maxis[col])
            self.minis[col] = min(chunk_df[col].min(), self.minis[col])

        chunk_df = chunk_df.sample(frac=1).reset_index(drop=True)

        out = chunk_df[self.coord_cols].values.T
        out = Points(
            X=out.astype(np.float32),
            track=(chunk_df[['track']] >= 0).values.squeeze(-1).astype(np.float32)
        )
        self.mean_hits.append(len(chunk_df))
        return TransformedPointsDataChunk(out, output_name)

    def forward_fields(self):
        return {col: self.__dict__[col] for col in self._forward_fields}

    def reduce_fields(self, params_dict):
        for col in self.stats_cols:
            self.maxis[col] = max(self.maxis[col], params_dict['maxis'][col])
            self.minis[col] = min(self.minis[col], params_dict['minis'][col])

        self.mean_hits.extend(params_dict['mean_hits'])
        self.mean_tracks.extend(params_dict['mean_tracks'])
        self.mean_len_tracks.extend(params_dict['mean_len_tracks'])


    def save_on_disk(self,
                     processed_data: ProcessedPointsData):
        LOGGER.info("\n\n=================\nSome stats:")
        for col in self.stats_cols:
            LOGGER.info(f'MAXIS {col}: {self.maxis[col]}')
            LOGGER.info(f'MINIS {col}: {self.minis[col]}')
        LOGGER.info(f'Mean hits per event: {np.mean(self.mean_hits)}')
        LOGGER.info(f'Mean tracks per event: {np.mean(self.mean_tracks)}')
        LOGGER.info(f'Mean len_tracks per event: {np.mean(self.mean_len_tracks)}')
        LOGGER.info('End\n===============\n')
        super(PointNet_ProcessorBMN7, self).save_on_disk(processed_data)


@gin.configurable
class PointNet_ProcessorBMN7_dist(PointNet_ProcessorBMN7):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms
        )
        raise NotImplementedError('implements cols for stats plz')

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return TransformedPointsDataChunk(None, '')

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'points_{idx}_{chunk_id}')

        for col in ['x', 'y', 'z']:
            self.maxis[col] = max(chunk_df[col].max(), self.maxis[col])
            self.minis[col] = min(chunk_df[col].min(), self.minis[col])

        chunk_df = chunk_df[chunk_df.det == 1]
        if chunk_df.empty:
            LOGGER.warning(f'SKIPPED broken {chunk_id} event')
            return TransformedPointsDataChunk(None, output_name)

        track_pnts = chunk_df[chunk_df.track >= 0]
        xs = track_pnts[['x']].values
        ys = track_pnts[['y']].values
        zs = track_pnts[['z']].values

        a = chunk_df.apply(lambda pnt: np.min(
            np.linalg.norm(
                np.hstack((
                    (xs - pnt.x, ys - pnt.y, zs - pnt.z)
                )), axis=-1)), axis=1)

        chunk_df['dist'] = a

        out = chunk_df[['x', 'y', 'z']].values.T
        out = Points(
            X=out.astype(np.float32),
            track=chunk_df[['dist']].values.squeeze(-1).astype(np.float32)
        )
        self.mean_hits.append(len(chunk_df))
        return TransformedPointsDataChunk(out, output_name)


@gin.configurable
class PointNet_ProcessorBMN7_impulse(PointNet_ProcessorBMN7):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stats_cols,
                 impulses,
                 transforms: List[BaseTransformer] = None):
        super().__init__(
            output_dir=output_dir,
            data_df=data_df,
            stats_cols=stats_cols,
            transforms=transforms
        )
        self.impulse_minmax = impulses

    def preprocess_chunk(self,
                         chunk: PointsDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return TransformedPointsDataChunk(None, '')

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'points_{idx}_{chunk_id}')
        chunk_df = chunk_df[chunk_df.det == 1]

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.info(f'\nPointNet_Processor got hit with id < -1!\n for event {chunk_id}')
            return TransformedPointsDataChunk(None, '')

        if chunk_df[chunk_df.track == -1].empty:
            LOGGER.info(f'\nPointNet_Processor got event with no fakes!\n for event {chunk_id}')
            return TransformedPointsDataChunk(None, '')

        if chunk_df[chunk_df.track != -1].empty:
            LOGGER.info(f'\nPointNet_Processor got event with no real hits!\n for event {chunk_id}')
            return TransformedPointsDataChunk(None, '')

        for key, val in self.impulse_minmax.items():
            vals = chunk_df[[key]].values
            normed = vals / val[1]
            chunk_df[key] = normed
            true = chunk_df[chunk_df.track >= 0][key].values
            false = chunk_df[chunk_df.track < 0][key].values
            self.maxis[key + "_true"] = max(self.maxis[key + "_true"], true.max())
            self.minis[key + "_true"] = min(self.minis[key + "_true"], true.min())
            self.maxis[key + "_false"] = max(self.maxis[key + "_false"], false.max())
            self.minis[key + "_false"] = min(self.minis[key + "_false"], false.min())

        chunk_df['station'] = (chunk_df['station'].values / 5.0)

        for col in self.stats_cols:
            if '_' in col:
                continue
            self.maxis[col] = max(chunk_df[col].max(), self.maxis[col])
            self.minis[col] = min(chunk_df[col].min(), self.minis[col])

        # track_pnts = chunk_df[chunk_df.track >= 0]
        # xs = track_pnts[['x']].values
        # ys = track_pnts[['y']].values
        # zs = track_pnts[['z']].values
        # stats = track_pnts[['station']].values / 7.0

        out = chunk_df[['x', 'y', 'z', 'station']].values.T
        # if not chunk_df[(chunk_df.track != -1) & (chunk_df.track < 0) ].empty:
        #    LOGGER.info("\nPointNet_Processor got hit with id < -1!\n for event %d" % chunk_id)
        #    return TransformedPointsDataChunk(None, "")

        tgt = chunk_df[['px', 'py', 'pz']].values
        tgt_hit = (chunk_df['track'].values != -1).astype(np.float32).reshape(-1, 1)
        tgt = np.hstack((tgt_hit, tgt))

        out = Points(
            X=out.astype(np.float32),
            track=tgt
        )
        self.mean_hits.append(len(chunk_df))
        return TransformedPointsDataChunk(out, output_name)
