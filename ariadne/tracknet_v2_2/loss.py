import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss,BCELoss
gin.external_configurable(BCELoss)


@gin.configurable()
class TrackNetV22BCELoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input, target):
        return super().forward(input, target.unsqueeze(-1))


