import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class PointNetWeightedBCE(nn.Module):
    """Weighted binary cross entropy for RDGraphNet"""
    def __init__(self, real_weight, fake_weight):
        super().__init__()
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def forward(self, preds, target):
        # Compute target weights on-the-fly for loss function
        batch_weights_real = target * self.real_weight
        batch_weights_fake = (1 - target) * self.fake_weight
        batch_weights = batch_weights_real + batch_weights_fake
        # compute loss with weights
        return F.binary_cross_entropy(preds, target, weight=batch_weights)
        