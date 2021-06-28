import torch 
import torch.nn as nn
import torch.nn.functional as F
import gin

from ariadne.lightning import *
from ariadne.tracknet_v2.loss import *
from ariadne.data_loader import *
from ariadne.vertex_loot.dataset import *

# from lightning import *
# from tracknet_v2.loss import *
# from data_loader import *
# from vertex_loot.dataset import *

@gin.configurable
class UnetLootDataLoader(BaseDataLoader):

    def __init__(self,
                 batch_size: int,
                 dataset: TrackDataSet.__class__,
                 valid_size: float,
                 max_size=None,
                 with_random=True):
        super(UnetLootDataLoader, self).__init__(batch_size)
        self.dataset = dataset
        data_size = len(self.dataset)
        # n_valid = int(data_size * valid_size)
        n_valid = valid_size
        n_train = data_size - n_valid
        if with_random:
            self.train_data, self.val_data = random_split(self.dataset, [n_train, n_valid])
        else:
            self.train_data = Subset(self.dataset, range(0, n_train)),
            self.val_data = Subset(self.dataset, range(n_train, n_train + n_valid))

    def get_val_dataloader(self) -> DataLoader:
        return DataLoader(
                          dataset=self.val_data,
                          batch_size=self.batch_size,
                          num_workers=8
                          )

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=8
            )