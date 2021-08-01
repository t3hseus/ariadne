import gin
import pandas as pd

from torch.utils.data import Dataset

from ariadne.graph_net.dataset import SubsetWithItemLen
from ariadne_v2.jit_cacher import Cacher
from experiments.graph.graph_utils.graph import sparse_to_graph
from experiments.graph.inferrer import GraphDataset


@gin.configurable
class TorchGraphDataset(GraphDataset, Dataset):

    def __init__(self, dataset_name, cache_graphs=False, override_len=None):
        super(TorchGraphDataset, self).__init__(dataset_name)
        self.cache_graphs = cache_graphs
        self.graphs = {}
        self.len_override = override_len

    def __getitem__(self, index):
        graph = self.graphs.get(index, None)
        if graph is None:
            dataset_path = self.events_db_df.iloc[index]['name']
            graph_dict_hdf5 = self.get(dataset_path)
            graph = sparse_to_graph(**{col: val[()] for col, val in graph_dict_hdf5.items()})
            if self.cache_graphs:
                self.graphs[index] = graph
        return graph

    def connect(self, cacher: Cacher, dataset_path: str, drop_old: bool, mode: str = None):
        super().connect(cacher, dataset_path, drop_old, mode=mode)
        self.events_db_df: pd.DataFrame = self.meta.get_df(self.INFO_DF_NAME)

    def _disconnect(self):
        super().disconnect()

    def __len__(self):
        assert self.connected
        if self.len_override:
            return self.len_override
        return len(self.events_db_df)

    def get_item_length(self, item_index):
        assert self.connected
        return self.events_db_df.iloc[item_index]['X']


@gin.configurable(allowlist=[])
class SubsetTorchGraphDataset(SubsetWithItemLen):
    def __init__(self, dataset: TorchGraphDataset, indices):
        super(SubsetWithItemLen, self).__init__(dataset, indices)

    def get_item_length(self, item_index):
        return self.dataset.get_item_length(self.indices[item_index])
