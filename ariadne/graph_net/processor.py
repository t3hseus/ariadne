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
from ariadne.graph_net.graph_utils.graph import Graph, save_graphs_new
from ariadne.graph_net.graph_utils.graph_prepare_utils import (
    #to_pandas_graph_from_df,
    #get_pd_line_graph,
    #apply_edge_restriction,
    #construct_output_graph,
    construct_output_graph_new,
    get_plain_graph,
    get_plain_graph_new,
    get_edges_from_supernodes_new,
    apply_edge_rescriction_new,
    apply_nodes_restrictions
)

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable(denylist=['df_chunk_data'])
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


@gin.configurable(denylist=['data_df'])
class GraphNet_Processor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 df_suffixes: Tuple[str],
                 transforms: List[BaseTransformer] = None,
                 **kwargs):
        super().__init__(
            processor_name='RDGraphNet_v1_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms
        )
        self.func_vals = kwargs
        self._suffixes_df = df_suffixes

    def generate_chunks_iterable(self) -> Iterable[GraphDataChunk]:
        return self.data_df.groupby('event')

    def total_chunks(self) -> int:
        return self.data_df['event'].nunique()

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> GraphDataChunk:
        processed = self.transformer(chunk_df)
        return GraphDataChunk(processed)

    '''
    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return ReversedDiGraphDataChunk(None, '')

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'graph_{idx}_{chunk_id}')

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.warning(f'SKIPPED {chunk_id} event with bad indices')
            return ReversedDiGraphDataChunk(None, output_name)

        chunk_df = chunk_df[['event', 'r', 'phi', 'z', 'station', 'track']]
        pd_graph_df = to_pandas_graph_from_df(
            single_event_df=chunk_df,
            suffixes=self._suffixes_df,
            compute_is_true_track=True
        )

        nodes_t, edges_t = get_pd_line_graph(
            pd_graph_df, apply_nodes_restrictions,
            **self.func_vals['get_pd_line_graph'],
            spec_kwargs=self.func_vals['get_supernodes_df']
        )

        # skip completely broken events (happens rarely but should not disturb the whole preprocessing)
        if edges_t.empty:
            LOGGER.warning(f'SKIPPED broken {chunk_id} event')
            return ReversedDiGraphDataChunk(None, output_name)

        # here we are filtering out superedges, trying to leave true superedges as much as we can
        edges_filtered = apply_edge_restriction(edges_t,  **self.func_vals['apply_edge_restriction'])

        # construct the output graph suitable for the neural network
        out = construct_output_graph(
            hits=nodes_t,
            edges=edges_filtered,
            feature_names=['y_p', 'y_c', 'z_p', 'z_c', 'z'],
            feature_scale=[1., 1., 1., 1., 1.],
        )
        return ReversedDiGraphDataChunk(out, output_name)
    '''

    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return ReversedDiGraphDataChunk(None, '')

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'graph_{idx}_{chunk_id}')

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.warning(f'SKIPPED {chunk_id} event with bad indices')
            return ReversedDiGraphDataChunk(None, output_name)

        suff_df = ('_p', '_c')
        chunk_df['index_old'] = chunk_df.index
        event = chunk_df[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]
        #
        device = 'cpu'
        event_data = torch.from_numpy(event.to_numpy()).to(device)
        column_index = dict(zip(event.columns, list(range(0, len(event.columns)))))

        convert(event_data)

        nodes, edges, node_indices, edge_indices = get_plain_graph_new(event_data, column_index, device=device,
                                                                       suffixes=suff_df, compute_is_true_track=True,
                                                                       restrictions_0=
                                                                       self.func_vals['get_pd_line_graph'][
                                                                           'restrictions_0'],
                                                                       restrictions_1=
                                                                       self.func_vals['get_pd_line_graph'][
                                                                           'restrictions_1'])
        node_index = node_indices['index_old']

        constraints = torch.tensor(self.func_vals['normalize']['constraints'])

        normalize(nodes, [node_indices[x] for x in ['x', 'y', 'z']], constraints=constraints, mode='node')

        normalize(edges, [edge_indices['dx'], edge_indices['dy'], edge_indices['dz']], constraints=constraints,
                  mode='edge')

        edges = apply_edge_rescriction_new(edges, edge_indices,
                                           restrictions_0=self.func_vals['get_pd_line_graph']['restrictions_0'],
                                           restrictions_1=self.func_vals['get_pd_line_graph']['restrictions_1'])


        out = construct_output_graph_new(nodes, edges, \
                                         torch.tensor([node_indices[x] for x in ['y', 'y', 'z']]), \
                                         1., node_index, edge_indices['is_true'], edge_indices['index_old_p'],
                                         edge_indices['index_old_c'])

        return ReversedDiGraphDataChunk(out, output_name)

    def postprocess_chunks(self,
                           chunks: List[ReversedDiGraphDataChunk]) -> ProcessedGraphData:
        return ProcessedGraphData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedGraphData):
        save_graphs_new(processed_data.processed_data)

import torch
from to_torchscript import normalize,convert

@gin.configurable(denylist=['data_df'])
class RDGraphNet_Processor(DataProcessor):
    def __init__(self,
                 output_dir: str,
                 data_df: pd.DataFrame,
                 df_suffixes: Tuple[str],
                 transforms: List[BaseTransformer] = None,
                 **kwargs):
        super().__init__(
            processor_name='RDGraphNet_v1_Processor',
            output_dir=output_dir,
            data_df=data_df,
            transforms=transforms
        )
        self.func_vals = kwargs
        self._suffixes_df = df_suffixes

    def generate_chunks_iterable(self) -> Iterable[GraphDataChunk]:
        return self.data_df.groupby('event')

    def total_chunks(self) -> int:
        return self.data_df['event'].nunique()

    def construct_chunk(self,
                        chunk_df: pd.DataFrame) -> GraphDataChunk:
        processed = self.transformer(chunk_df)
        return GraphDataChunk(processed)

    def preprocess_chunk(self,
                         chunk: GraphDataChunk,
                         idx: str) -> ProcessedDataChunk:
        chunk_df = chunk.df_chunk_data

        if chunk_df.empty:
            return ReversedDiGraphDataChunk(None, '')

        chunk_id = int(chunk_df.event.values[0])
        output_name = os.path.join(self.output_dir, f'graph_{idx}_{chunk_id}')

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.warning(f'SKIPPED {chunk_id} event with bad indices')
            return ReversedDiGraphDataChunk(None, output_name)

        suff_df = ('_p', '_c')
        chunk_df['index_old'] = chunk_df.index
        event = chunk_df[['event', 'x', 'y', 'z', 'station', 'track', 'index_old']]
        #
        device = 'cpu'
        event_data = torch.from_numpy(event.to_numpy()).to(device)
        column_index = dict(zip(event.columns, list(range(0, len(event.columns)))))

        convert(event_data)

        nodes, edges, node_indices, edge_indices = get_plain_graph_new(event_data, column_index, device=device,
                                                                       suffixes=suff_df, compute_is_true_track=True,
                                                                       restrictions_0 = self.func_vals['get_pd_line_graph']['restrictions_0'],
                                                                       restrictions_1 = self.func_vals['get_pd_line_graph']['restrictions_1'])

        constraints =torch.tensor( self.func_vals['normalize']['constraints'] )

        normalize(nodes, [node_indices[x] for x in ['x', 'y', 'z']], constraints=constraints,mode='node')

        normalize(edges, [edge_indices['x_p'], edge_indices['y_p'], edge_indices['z_p']], constraints=constraints,mode='node')
        normalize(edges, [edge_indices['x_c'], edge_indices['y_c'], edge_indices['z_c']], constraints=constraints,mode='node')

        normalize(edges, [edge_indices['dx'], edge_indices['dy'], edge_indices['dz']], constraints=constraints,mode='edge')

        edges = apply_edge_rescriction_new(edges, edge_indices,restrictions_0=self.func_vals['get_pd_line_graph']['restrictions_0'],
                                           restrictions_1=self.func_vals['get_pd_line_graph']['restrictions_1'])

        nodes = edges
        node_indices = edge_indices
        node_index = edge_indices['index']

        edges, edge_indices = get_edges_from_supernodes_new(edges, edge_indices)

        out = construct_output_graph_new(nodes, edges, \
                                   torch.tensor([node_indices[x] for x in ['y_p', 'y_c', 'z_p', 'z_c', 'z'] ]), \
                                   1., node_index, edge_indices['is_true'], edge_indices['index_old_p'], edge_indices['index_old_c'])

        return ReversedDiGraphDataChunk(out, output_name)

    def postprocess_chunks(self,
                           chunks: List[ReversedDiGraphDataChunk]) -> ProcessedGraphData:
        return ProcessedGraphData(chunks)


    def save_on_disk(self,
                     processed_data: ProcessedGraphData):
        save_graphs_new(processed_data.processed_data)