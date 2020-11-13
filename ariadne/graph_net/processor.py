import logging
from typing import List, Tuple, Optional, Iterable

from ariadne.graph_net.graph_utils.graph import Graph, save_graphs_new, Graph_V2, save_graph_v2
from ariadne.graph_net.graph_utils.graph_prepare_utils import to_pandas_graph_from_df, get_pd_line_graph, \
    apply_edge_restriction, construct_output_graph, apply_nodes_restrictions, edge_list_from_ind
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


class EdgeListGraphDataChunk(ProcessedDataChunk):

    def __init__(self,
                 processed_object: Optional[Graph_V2],
                 output_name: str):
        super().__init__(processed_object)
        self.processed_object = processed_object
        self.output_name = output_name


class ProcessedGraphData(ProcessedData):

    def __init__(self, processed_data: List[ReversedDiGraphDataChunk]):
        super().__init__(processed_data)
        self.processed_data = processed_data


@gin.configurable(blacklist=['data_df', 'processor_name'])
class GraphNet_Processor(DataProcessor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints,
                 df_suffixes: Tuple[str],
                 processor_name: str = 'RDGraphNet_v1_Processor'
                 ):
        super().__init__(
            processor_name=processor_name,
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

    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/graph_%s_%d' % (idx, chunk_id))
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


@gin.configurable(blacklist=['data_df', 'processor_name'])
class GraphNet_V2_Processor(GraphNet_Processor):

    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 stations_constraints,
                 df_suffixes: Tuple[str],
                 n_stations: int,
                 processor_name: str = 'GraphNet_V2_Processor'
                 ):
        super().__init__(
            processor_name=processor_name,
            output_dir=output_dir,
            data_df=data_df,
            df_suffixes=df_suffixes,
            stations_constraints=stations_constraints)
        self.n_stations = n_stations

    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data.reset_index()
        chunk_id = int(chunk_df.event.values[0])
        output_name = (self.output_dir + '/el_graph_%s_%d' % (idx, chunk_id))

        indices = [None] * self.n_stations
        for station_idx, stat in chunk_df.groupby('station'):
            indices[station_idx] = stat.index.values
        edge_index = edge_list_from_ind(indices[0], indices[1])

        for station_idx in range(2, self.n_stations):
            edge_index = np.concatenate((edge_index,
                                         edge_list_from_ind(
                                             indices[station_idx - 1],
                                             indices[station_idx]
                                         )), axis=1)
        node_features = chunk_df[['r', 'phi', 'z']].values
        y = chunk_df['track'].values
        y[y != -1] = 1
        y[y == -1] = 0
        assert (len(y) == len(node_features) and 0 < len(y) and len(y) >= edge_index[0].max())
        out = Graph_V2(X=node_features, edge_index=edge_index, y=y)
        return EdgeListGraphDataChunk(out, output_name)

    def save_on_disk(self,
                     processed_data: ProcessedGraphData):
        for graph in processed_data.processed_data:
            if graph.processed_object is None:
                continue
            processed_graph: Graph = graph.processed_object
            save_graph_v2(processed_graph, graph.output_name)
