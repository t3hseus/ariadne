import os

import gin
import torch
from torch.utils.data import Dataset

from ariadne.graph_net.graph_utils.graph import load_graph


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


# TODO: rewrite it
@gin.configurable('graph_collate_fn')
def collate_fn(graphs):
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
        batch_target = torch.from_numpy(g.y[None]).float()
        return batch_inputs, batch_target

    assert False

