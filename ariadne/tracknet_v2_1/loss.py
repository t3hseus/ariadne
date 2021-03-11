import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

gin.external_configurable(CrossEntropyLoss)
@gin.configurable()
class TrackNetCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean'):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(weight=weight,
                     size_average=size_average,
                     ignore_index=ignore_index,
                     reduce=reduce,
                     reduction=reduction)




