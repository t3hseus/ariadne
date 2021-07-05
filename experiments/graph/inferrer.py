import logging
import os
import pathlib
from typing import Tuple, Dict, Any

import pandas as pd

from ariadne_v2.dataset import AriadneDataset
from ariadne_v2.inference import IPreprocessor, IPostprocessor
from ariadne_v2.jit_cacher import Cacher
from ariadne_v2.preprocessing import DFDataChunk
from .graph_utils.graph import save_graphs_new, save_graph, save_graph_hdf5, graph_to_sparse
from .graph_utils.graph_prepare_utils import to_pandas_graph_from_df, get_pd_line_graph, apply_nodes_restrictions, \
    apply_edge_restriction, construct_output_graph

LOGGER = logging.getLogger('ariadne.prepare')


class GraphPreprocessor(IPreprocessor):
    _suffixes_df = ('_p', '_c')
    kwargs = None

    def __init__(self,
                 _suffixes_df: Tuple[str],
                 kwargs: Dict):
        self._suffixes_df = _suffixes_df
        self.kwargs = kwargs

    def __call__(self, chunk: DFDataChunk):
        chunk_df = chunk.as_df()

        if chunk_df.empty:
            return None

        chunk_id = int(chunk_df.event.values[0])

        if not chunk_df[chunk_df.track < -1].empty:
            LOGGER.warning(f'SKIPPED {chunk_id} event with bad indices')
            return None

        chunk_df = chunk_df[['event', 'r', 'phi', 'z', 'station', 'track']]
        pd_graph_df = to_pandas_graph_from_df(
            single_event_df=chunk_df,
            suffixes=self._suffixes_df,
            compute_is_true_track=True
        )

        nodes_t, edges_t = get_pd_line_graph(
            pd_graph_df, apply_nodes_restrictions,
            **self.kwargs['get_pd_line_graph'],
            spec_kwargs=self.kwargs['get_supernodes_df']
        )

        # skip completely broken events (happens rarely but should not disturb the whole preprocessing)
        if edges_t.empty:
            LOGGER.warning(f'SKIPPED broken {chunk_id} event')
            return None

        # here we are filtering out superedges, trying to leave true superedges as much as we can
        edges_filtered = apply_edge_restriction(edges_t, **self.kwargs['apply_edge_restriction'])
        return DFDataChunk.from_df(nodes_t), DFDataChunk.from_df(edges_filtered)


class SaveGraphs(IPostprocessor):

    def __call__(self, out_dir: str, out: Tuple, idx: str):
        if out is None:
            return False
        ret = construct_output_graph(
            hits=out[0].as_df(),
            edges=out[1].as_df(),
            feature_names=['y_p', 'y_c', 'z_p', 'z_c', 'z'],
            feature_scale=[1., 1., 1., 1., 1.],
        )
        save_graph(ret, os.path.join(out_dir, idx))
        return True


class SaveGraphsHDF5(IPostprocessor):

    def __call__(self, out_dir, out: Tuple, idx: str):
        if out is None:
            return False
        ret = construct_output_graph(
            hits=out[0].as_df(),
            edges=out[1].as_df(),
            feature_names=['y_p', 'y_c', 'z_p', 'z_c', 'z'],
            feature_scale=[1., 1., 1., 1., 1.],
        )

        save_graph_hdf5(out_dir, ret, idx)
        return True



class GraphDataset(AriadneDataset):

    DATA_COLUMNS = ["X", "y", "Ri_rows", "Ro_rows", "Ri_cols", "Ro_cols"]

    def __init__(self, dataset_path):
        super(GraphDataset, self).__init__(dataset_path)

        self.infos = {col:[] for col in ["name"] + self.DATA_COLUMNS}

    def add(self, key, values: Dict):
        super().add(key, values)
        self.infos["name"].append(key)
        for k, v in values.items():
            self.infos[k].append(v.shape[0])

    def _submit_local_data(self, prefix):
        def update(df):
            df = df if df is not None else pd.DataFrame()
            new_df = pd.DataFrame.from_dict(self.infos)
            return pd.concat([df, new_df], ignore_index=True)
        self.meta.update_df('shape0', update)


class SaveGraphsToDataset(IPostprocessor):

    def __call__(self, out:Any, ds:GraphDataset, idx: str):
        if out is None:
            return False

        ret = construct_output_graph(
            hits=out[0].as_df(),
            edges=out[1].as_df(),
            feature_names=['y_p', 'y_c', 'z_p', 'z_c', 'z'],
            feature_scale=[1., 1., 1., 1., 1.],
        )
        #save_graph(ret, os.path.join(ds.temp_dir, idx))
        ds.add(idx, graph_to_sparse(ret))

        return True