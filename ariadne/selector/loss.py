import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class SelectorWeightedCE(nn.Module):
    """Weighted binary cross entropy for RDGraphNet"""
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.Tensor(weights).cuda()

    def forward(self, preds, target):
        # compute loss with weights
        return F.cross_entropy(preds, target, weight=self.weights)
