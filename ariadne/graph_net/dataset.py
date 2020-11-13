import os
from typing import Iterable, List

import gin
import torch
import numpy as np

from torch.utils.data import Dataset
from ariadne.graph_net.graph_utils.graph import load_graph, Graph, Graph_V2, load_graph_v2


@gin.configurable
class GraphDataset(Dataset):

    def __init__(self, input_dir, n_samples=None):
        self.input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.npz')]
        self.filenames = (filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        # uncomment to find bad events in the dataset:
        # print(index, self.filenames[index])
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


@gin.configurable
class GraphDatasetFromMemory(Dataset):
    def __init__(self, input_graphs, n_samples=None):
        self.graphs = (input_graphs[:n_samples] if n_samples is not None else input_graphs)

    def __getitem__(self, index):
        graph_with_ind = self.graphs[index]
        return Graph(graph_with_ind.X, graph_with_ind.Ri, graph_with_ind.Ro, graph_with_ind.y)

    def __len__(self):
        return len(self.graphs)


@gin.configurable('graph_collate_fn')
def collate_fn(graphs):
    """
    Collate function for building mini-batches from a list of hit-graphs.
    This function should be passed to the pytorch DataLoader.
    It will stack the hit graph matrices sized according to the maximum
    sizes in the batch and padded with zeros.
    This implementation could probably be optimized further.
    """
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
        batch_target = torch.from_numpy(g.y[None]).float()
        return {"inputs" : batch_inputs}, batch_target

    # Get the matrix sizes in this batch
    n_features = graphs[0].X.shape[1]
    n_nodes = np.array([g.X.shape[0] for g in graphs])
    n_edges = np.array([g.y.shape[0] for g in graphs])
    max_nodes = n_nodes.max()
    max_edges = n_edges.max()

    # Allocate the tensors for this batch
    batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    batch_Ri = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_Ro = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_y = np.zeros((batch_size, max_edges), dtype=np.float32)

    # Loop over samples and fill the tensors
    for i, g in enumerate(graphs):
        batch_X[i, :n_nodes[i]] = g.X
        batch_Ri[i, :n_nodes[i], :n_edges[i]] = g.Ri
        batch_Ro[i, :n_nodes[i], :n_edges[i]] = g.Ro
        batch_y[i, :n_edges[i]] = g.y

    batch_inputs = [torch.from_numpy(bm) for bm in [batch_X, batch_Ri, batch_Ro]]
    batch_target = torch.from_numpy(batch_y)
    return {"inputs" : batch_inputs}, batch_target

@gin.configurable
class Graph_V2_Dataset(GraphDataset):
    def __init__(self, input_dir, n_samples=None):
        super(Graph_V2_Dataset, self).__init__(input_dir=input_dir,
                                               n_samples=n_samples)

    def __getitem__(self, index):
        # uncomment to find bad events in the dataset:
        # print(index, self.filenames[index])
        return load_graph_v2(self.filenames[index])


@gin.configurable('graph_v2_collate_fn')
def collate_fn_v2(graphs: List[Graph_V2]):
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.tensor(g.X.astype(np.float32)), torch.tensor(g.edge_index)]
        batch_target = torch.tensor(g.y.astype(np.float32))
        return {"inputs" : batch_inputs}, batch_target
    assert False, "not supported"