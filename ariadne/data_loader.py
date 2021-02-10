from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader


class BaseDataLoader(metaclass=ABCMeta):

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_val_dataloader(self) -> DataLoader:
        pass

#     #needed for graph in tensorboard
#     @abstractmethod
#     def get_one_sample(self) -> dict:
#         pass