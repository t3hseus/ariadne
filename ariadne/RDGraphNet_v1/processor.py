import logging
from typing import List, Tuple

from RDGraphNet_v1.graph_utils.graph_prepare_utils import to_pandas_graph_from_df
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData

import gin
import pandas as pd

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(blacklist=['df_chunk_data'])
class GraphDataChunk(DataChunk):

    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)


@gin.configurable(blacklist=['data_df'])
class RDGraphNet_v1_Processor(DataProcessor):

    def __init__(self,
                 some_custom_field: str,
                 data_df: pd.DataFrame,
                 df_suffixes: Tuple[str] = ('_prev', '_current'),
                 ):
        super().__init__(
            processor_name='RDGraphNet_v1_Processor',
            data_df=data_df)
        self._suffixes_df = df_suffixes
        LOGGER.info("GOT %s \n%s \n========", data_df, some_custom_field)

    @gin.configurable(module='RDGraphNet_v1_Processor', blacklist=['chunk','idx'])
    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: int) -> List[ProcessedDataChunk]:
        pd_graph_df = to_pandas_graph_from_df(single_event_df=chunk.df_chunk_data,
                                              suffixes=self._suffixes_df,
                                              compute_is_true_track=True)

        pass

    def postprocess_chunk(self,
                          chunks: List[ProcessedDataChunk]) -> ProcessedData:
        pass
