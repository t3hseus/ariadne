import logging
from collections import OrderedDict
from typing import Collection, Dict

import gin

import os
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, Sampler, RandomSampler, Subset
from tqdm import tqdm

from ariadne.point_net.point.points import load_points, Points


class MemoryDataset(Dataset):

    def __init__(self, input_dir):
        super(MemoryDataset, self).__init__()
        self.input_dir = input_dir

    def load_data(self):
        raise NotImplementedError


class ItemLengthGetter(object):
    def __init__(self):
        pass

    def get_item_length(self, item_index):
        raise NotImplementedError


@gin.configurable
class PointsDatasetMemory(MemoryDataset, ItemLengthGetter):

    def __init__(self, input_dir, n_samples=None, pin_mem=False):
        super().__init__(os.path.expandvars(input_dir))
        self.n_samples = n_samples
        self.filenames = []
        self.points = {}
        self.pin_mem = pin_mem

        filenames = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.npz')]
        self.filenames = (filenames[:self.n_samples] if self.n_samples is not None else filenames)

    def get_item_length(self, item_index):
        item = None
        if self.pin_mem:
            if item_index in self.points:
                item = self.points[item_index]
            else:
                item = self.points[item_index] = load_points(self.filenames[item_index])
        else:
            item = load_points(self.filenames[item_index])
        return len(item.track)

    def __getitem__(self, index):
        # uncomment to find bad events in the dataset:
        # print(index, self.filenames[index])
        if self.pin_mem:
            if index in self.points:
                item = self.points[index]
            else:
                item = self.points[index] = load_points(self.filenames[index])
            return item
        return load_points(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


class SubsetWithItemLen(Subset, ItemLengthGetter):
    def __init__(self, dataset, indices):
        super(SubsetWithItemLen, self).__init__(dataset, indices)

    def get_item_length(self, item_index):
        return len(self.dataset[self.indices[item_index]].track)


@gin.configurable(blacklist=['data_source'])
class BatchBucketSampler(Sampler):
    def __init__(self, data_source: SubsetWithItemLen,
                 zero_pad_available,
                 batch_size,
                 drop_last,
                 shuffle):
        super(BatchBucketSampler, self).__init__(data_source)
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


@gin.configurable('points_collate_fn')
def collate_fn(points: Collection[Points]):
    batch_size = len(points)
    n_feat = 3

    n_dim = np.array([(p.X.shape[1]) for p in points])
    # print(n_dim)
    max_dim = n_dim.max()
    batch_inputs = np.zeros((batch_size, n_feat, max_dim), dtype=np.float32)
    batch_targets = np.zeros((batch_size, max_dim), dtype=np.float32)

    for i, p in enumerate(points):
        batch_inputs[i, :, :n_dim[i]] = p.X

        batch_targets[i, :n_dim[i]] = p.track

    return {"x": torch.from_numpy(batch_inputs)}, torch.from_numpy(batch_targets)


@gin.configurable('points_dist_collate_fn')
def collate_fn_dist(points: Collection[Points]):
    batch_size = len(points)
    n_feat = 3

    n_dim = np.array([(p.X.shape[1]) for p in points])
    # print(n_dim)
    max_dim = n_dim.max()
    batch_inputs = np.zeros((batch_size, n_feat, max_dim), dtype=np.float32)
    batch_targets = np.full((batch_size, max_dim), 0.5, dtype=np.float32)

    for i, p in enumerate(points):
        batch_inputs[i, :, :n_dim[i]] = p.X

        batch_targets[i, :n_dim[i]] = p.track

    return {"x": torch.from_numpy(batch_inputs)}, torch.from_numpy(batch_targets)
