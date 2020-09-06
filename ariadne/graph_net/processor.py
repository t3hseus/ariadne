import logging
from typing import List, Tuple, Optional, Iterable

from ariadne.graph_net.graph_utils.graph import Graph, save_graphs_new
from ariadne.graph_net.graph_utils.graph_prepare_utils import to_pandas_graph_from_df, get_pd_line_graph, \
    apply_edge_restriction, construct_output_graph, apply_nodes_restrictions
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData

import gin
import pandas as pd
import numpy as np

from ariadne.transformations import Compose, StandardScale, ToCylindrical, ConstraintsNormalize

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(blacklist=['df_chunk_data'])
class GraphDataChunk(DataChunk):

    def __init__(self, df_chunk_data: pd.DataFrame):
        super().__init__(df_chunk_data)


class ReversedDiGraphDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: Optional[Graph],
                 output_name: str):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name


class ProcessedGraphData(ProcessedData):

    def __init__(self, processed_data: List[ReversedDiGraphDataChunk]):
        super().__init__(processed_data)
        self.processed_data = processed_data


@gin.configurable(blacklist=['data_df'])
class GraphNet_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints,
                 df_suffixes: Tuple[str]
                 ):
        super().__init__(
            processor_name='RDGraphNet_v1_Processor',
            output_dir=output_dir,
            data_df=data_df)

        self._suffixes_df = df_suffixes
        self.transformer = Compose([
            ConstraintsNormalize(
                use_global_constraints=False,
                constraints=stations_constraints
            ),
            ToCylindrical(drop_old=True, cart_columns=('y', 'x'))
        ])

    def generate_chunks_iterable(self) -> Iterable[GraphDataChunk]:
        return self.data_df.groupby('event')

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> GraphDataChunk:
        processed = self.transformer(chunk_df)
        return GraphDataChunk(processed)

    @gin.configurable(blacklist=['chunk', 'idx'])
    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: int) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/graph_%d' % chunk_id)
        pd_graph_df = to_pandas_graph_from_df(single_event_df=chunk_df,
                                              suffixes=self._suffixes_df,
                                              compute_is_true_track=True)
        nodes_t, edges_t = get_pd_line_graph(pd_graph_df, apply_nodes_restrictions)
        # skip completely broken events (happens rarely but should not disturb the whole preprocessing)
        if edges_t.empty:
            LOGGER.warning("SKIPPED broken %d event" % chunk_id)
            return ReversedDiGraphDataChunk(None, output_name)

        # here we are filtering out superedges, trying to leave true superedges as much as we can
        edges_filtered = apply_edge_restriction(edges_t)

        # construct the output graph suitable for the neural network
        out = construct_output_graph(nodes_t, edges_filtered, ['y_p', 'y_c', 'z_p', 'z_c', 'z'],
                                     [np.pi, np.pi, 1., 1., 1.], 'edge_index_p', 'edge_index_c')

        return ReversedDiGraphDataChunk(out, output_name)

    def postprocess_chunks(self,
                           chunks: List[ReversedDiGraphDataChunk]) -> ProcessedGraphData:
        return ProcessedGraphData(chunks)

    def save_on_disk(self,
                     processed_data: ProcessedGraphData):
        save_graphs_new(processed_data.processed_data)
