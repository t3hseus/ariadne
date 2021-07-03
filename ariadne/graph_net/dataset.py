import gin
from tqdm import tqdm
import os
import logging
import torch
import numpy as np
from collections import OrderedDict

from torch.utils.data import Dataset, BatchSampler, Sampler, RandomSampler, Subset
from ariadne.graph_net.graph_utils.graph import load_graph, Graph

LOGGER = logging.getLogger('ariadne.graph.dataset')

@gin.configurable
class GraphDataset(Dataset):

    def __init__(self, input_dir, n_samples=None):
        self.input_dir = os.path.expandvars(input_dir)
        LOGGER.info(f"[GraphDataset] Building dataset for folder '{self.input_dir}'")
        scanner = os.scandir(self.input_dir)
        if n_samples is None:
            max_count = 1_000_000
            LOGGER.info(f"[GraphDataset] n_samples was not set. using at most {n_samples} files.")
        else:
            max_count = n_samples
        self.filenames = [""] * max_count

        for idx in tqdm(range(max_count)):
            item = next(scanner, None)
            while item is not None and not item.name.endswith('.npz'):
                item = next(scanner, None)
            if item is None:
                if n_samples is not None:
                    raise RuntimeError(f"Run out of files. requested {n_samples} but {idx} was found. Exiting")
                else:
                    self.filenames = (self.filenames[:idx])
                    break
            self.filenames[idx] = os.path.join(self.input_dir, item.name)

        LOGGER.info(f"[GraphDataset] Total number of files: {len(self.filenames)}")



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


class MemoryDataset(GraphDataset):

    def __init__(self, input_dir, n_samples):
        super(MemoryDataset, self).__init__(input_dir=input_dir, n_samples=n_samples)

    def load_data(self):
        raise NotImplementedError


class ItemLengthGetter(object):
    def __init__(self):
        pass

    def get_item_length(self, item_index):
        raise NotImplementedError


@gin.configurable
class GraphsDatasetMemory(MemoryDataset, ItemLengthGetter):

    def __init__(self, input_dir, n_samples=None, pin_mem=False):
        super().__init__(os.path.expandvars(input_dir), n_samples=n_samples)
        self.pin_mem = pin_mem
        self.graphs = {}

    def get_item_length(self, item_index):
        item = None
        if self.pin_mem:
            if item_index in self.graphs:
                item = self.graphs[item_index]
            else:
                item = self.graphs[item_index] = load_graph(self.filenames[item_index])
        else:
            item = load_graph(self.filenames[item_index])
        return len(item.y)

    def __getitem__(self, index):
        # uncomment to find bad events in the dataset:
        # print(index, self.filenames[index])
        if self.pin_mem:
            if index in self.graphs:
                item = self.graphs[index]
            else:
                item = self.graphs[index] = load_graph(self.filenames[index])
            return item
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


class SubsetWithItemLen(Subset, ItemLengthGetter):
    def __init__(self, dataset, indices):
        super(SubsetWithItemLen, self).__init__(dataset, indices)

    def get_item_length(self, item_index):
        return len(self.dataset[self.indices[item_index]].y)


@gin.configurable(denylist=['data_source'])
class GraphBatchBucketSampler(Sampler):
    def __init__(self, data_source: SubsetWithItemLen,
                 zero_pad_available,
                 batch_size,
                 drop_last,
                 shuffle):
        super(GraphBatchBucketSampler, self).__init__(data_source)
        self.data_source = data_source
        self.zero_pad_available = zero_pad_available
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sampler = None

        self.buckets = OrderedDict()

        self.indices = []
        self._build_buckets()

    def _build_buckets(self):
        logging.info("_build buckets:")
        for key in tqdm(range(len(self.data_source))):
            len_ = self.data_source.get_item_length(key)
            if len_ not in self.buckets:
                self.buckets[len_] = [key]
            else:
                self.buckets[len_].append(key)

        if not self.zero_pad_available:
            raise NotImplementedError

            for len_, objs in self.buckets.items():
                if len(objs) < self.batch_size:
                    self.buckets[len_] = []

            self.buckets = {len_: objs[:len(objs) - (len(objs) % self.batch_size)]
                            for len_, objs in self.buckets.items() if len(objs) >= self.batch_size}
            self.indices = [[item for item in objs] for len_, objs in self.buckets.items()]

        logging.info("_build buckets end")
        self._store_indices()

    def _store_indices(self):
        self.indices = [[]]
        _last_inserted = 0
        values_iter = iter(self.buckets.values())
        while True:
            try:
                cur_array = next(values_iter)
                for item in cur_array:
                    self.indices[_last_inserted].append(item)
                    if len(self.indices[_last_inserted]) == self.batch_size:
                        self.indices.append([])
                        _last_inserted += 1
            except StopIteration:
                if len(self.indices[_last_inserted]) < self.batch_size:
                    self.indices = self.indices[:-1]
                break

        if self.shuffle and (len(self.indices) // 5) > 0:
            _new_indices = [[]]
            _idx = 0
            NEAREST = 5
            _last_inserted = 0
            while _idx < len(self.indices) // NEAREST:
                arr = []
                idx = _idx * NEAREST
                arr.extend(self.indices[idx])
                arr.extend(self.indices[idx + 1])
                arr.extend(self.indices[idx + 2])
                arr.extend(self.indices[idx + 3])
                arr.extend(self.indices[idx + 4])
                perm = iter(torch.randperm(len(arr), generator=None).tolist())
                for arr_idx in perm:
                    _new_indices[_last_inserted].append(arr[arr_idx])

                    if len(_new_indices[_last_inserted]) == self.batch_size:
                        _new_indices.append([])
                        _last_inserted += 1
                _idx += 1
            self.indices = _new_indices
            if len(self.indices[_last_inserted]) < self.batch_size:
                self.indices = self.indices[:-1]
            for batch in self.indices:
                assert len(batch) == self.batch_size
            logging.info("\nreshuffle indices\n")

    def __iter__(self):
        self._store_indices()
        perm = list(range(len(self.indices)))
        if self.shuffle:
            perm = iter(torch.randperm(len(self.indices), generator=None).tolist())

        for bucket_idx in perm:
            bucket = self.indices[bucket_idx]
            yield bucket

        if not self.drop_last:
            raise NotImplementedError

    def __len__(self):
        if self.drop_last:
            return len(self.indices)
        else:
            raise NotImplementedError


@gin.configurable('graph_collate_fn')
def collate_fn(graphs):
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
        batch_target = torch.from_numpy(g.y[None]).float()
        return {'inputs': batch_inputs}, batch_target

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
    return {'inputs': batch_inputs}, batch_target
