import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from typing import Optional

gin.external_configurable(CrossEntropyLoss)
@gin.configurable()
class TrackNetCrossEntropyLoss(nn.BCEWithLogitsLoss):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduction: str = 'sum',
                 pos_weight=None):
        if weight:
            weight = torch.tensor(weight)
        if pos_weight:
            pos_weight = torch.tensor(pos_weight)
        else:
            pos_weight = torch.tensor(1.)

        super().__init__(weight=weight,
                     size_average=size_average,
                     reduction=reduction,
                     pos_weight=pos_weight)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return super().forward(input, target.unsqueeze(-1).float())


@gin.configurable()
class FocalLoss(TrackNetCrossEntropyLoss):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True, weight=None, size_average=None, pos_weight=None):
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        super().__init__(weight=weight,
                         size_average=size_average,
                         reduction='none',
                         pos_weight=pos_weight)
        if pos_weight:
            pos_weight = torch.tensor(pos_weight)
        else:
            pos_weight = torch.tensor(1.)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, inputs, targets):
        BCE_loss = super().forward(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.sum(F_loss)
        else:
            return F_loss