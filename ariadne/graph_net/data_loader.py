from collections import Callable

import gin
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from ariadne.data_loader import BaseDataLoader
from ariadne.graph_net.dataset import GraphDataset

from torch_geometric.data import DataLoader as GDataLoader


@gin.configurable
class GraphDataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: GraphDataset.__class__,
                 collate_fn: Callable,
                 n_train: int,
                 n_valid: int,
                 with_random=True):
        super(GraphDataLoader, self).__init__(batch_size)
        self.dataset = dataset(n_samples=n_valid + n_train)
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))
        self.collate_fn = collate_fn

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn)


@gin.configurable
class GraphGeometricDataLoader(GraphDataLoader):
    def __init__(self,
                 batch_size: int,
                 dataset: GraphDataset.__class__,
                 collate_fn: Callable,
                 n_train: int,
                 n_valid: int,
                 with_random=True):
        super(GraphGeometricDataLoader, self).__init__(
            batch_size, dataset, collate_fn, n_train, n_valid, with_random)

    def get_val_dataloader(self) -> DataLoader:
        return GDataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size)

    def get_train_dataloader(self) -> DataLoader:
        return GDataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size)
