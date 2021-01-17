import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from ariadne.point_net.model import MIN_EPS_HOLDER


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


@gin.configurable
class PointNetWeightedDistloss(nn.Module):
    def __init__(self, lambda1, lambda2):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.MIN_EPS = MIN_EPS_HOLDER.MIN_EPS

    def forward(self, preds, target):

        batch_tgt_inv = 1 - target + self.MIN_EPS
        # we leave only very close points
        batch_tgt_inv = 1. / batch_tgt_inv

        batch_pred_inv = torch.clamp(1 - preds, 1e-5, 1) + self.MIN_EPS
        batch_pred_inv = 1. / batch_pred_inv

        return self.lambda1 * torch.norm(batch_pred_inv - batch_tgt_inv) + self.lambda2 * torch.norm(preds - target)