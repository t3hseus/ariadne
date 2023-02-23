import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class PerceiverWeightedBCE(nn.Module):
    """Weighted binary cross entropy for Perceiver (with masked outputs and targets) """

    def __init__(self, real_weight, fake_weight):
        super().__init__()
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def forward(self, preds, target):
        # compute loss with weights
        # change to bce with logits - more numerically stable
        mask = target['mask']
        loss_raw = F.binary_cross_entropy_with_logits(preds,
                               target['y'], reduction='none')
        # TODO rewrite to have one weight
        loss_raw[target['y'] == 1] *= self.real_weight
        loss_raw[target['y'] == 0] *= self.fake_weight
        loss_raw = loss_raw * mask.unsqueeze(-1)

        return loss_raw.sum() / mask.sum()


@gin.configurable
class PerceiverFocalLoss(nn.Module):
    """ Focal loss with masked outputs and targets """
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.MIN_EPS = 0.01

    def forward(self, preds, target):
        mask = target['mask']
        BCE_loss = F.binary_cross_entropy_with_logits(preds,
                                                      target['y'], reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        F_loss = F_loss * mask.unsqueeze(-1)
        return F_loss.sum() / mask.sum()