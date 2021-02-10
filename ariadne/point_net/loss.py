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
        self.MIN_EPS = 0.01

    def forward(self, preds, target):
        return torch.norm(preds - target)
        # return torch.abs(1 - torch.cosine_similarity(preds, target).sum()) * torch.norm(preds - target)
        # return (1/mul) * torch.norm(preds - target) + big + less
        # return torch.nn.functional.mse_loss(preds, target)#torch.pairwise_distance(preds, target).sum()


@gin.configurable
class PointNetWeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.MIN_EPS = 0.01

    def forward(self, preds, target):
        BCE_loss = F.binary_cross_entropy(preds, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


@gin.configurable
class PointNetImpulseLoss(nn.Module):
    """Weighted binary cross entropy for RDGraphNet"""

    def __init__(self, real_weight, fake_weight):
        super().__init__()
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def forward(self, preds_all, target_all):
        preds = preds_all[:, :, 0]
        target = target_all[:, :, 0]
        # Compute target weights on-the-fly for loss function
        batch_weights_real = target * self.real_weight
        batch_weights_fake = (1 - target) * self.fake_weight
        batch_weights = batch_weights_real + batch_weights_fake
        # compute loss with weights

        impulse_pred = preds_all[:, :,  1:]
        impulse_tgt = target_all[:, :, 1:]
        return F.binary_cross_entropy(preds, target, weight=batch_weights) * 0.5 + torch.norm(impulse_pred - impulse_tgt) * 0.5


@gin.configurable
class PointNetClassificationLoss(nn.Module):
    def __init__(self,
                 real_weight=1,
                 fake_weight=1,
                 is_softmax=False,
                 alpha=0.0001):
        super().__init__()
        self.real_weight = real_weight
        self.fake_weight = fake_weight
        self.alpha = alpha
        if is_softmax:
            self.criterion = F.cross_entropy
        else:
            self.criterion = F.binary_cross_entropy_with_logits

    def _regularization_term(self, transform_matrix):
        # identity matrix
        id_transform = torch.eye(
            transform_matrix.size(-1),
            requires_grad=True
        ).repeat(transform_matrix.size(0), 1, 1).to(transform_matrix.device)
        return torch.norm(
            id_transform - torch.bmm(transform_matrix, transform_matrix.transpose(1, 2))
        )

    def forward(self, preds, target):
        # Compute target weights on-the-fly for loss function
        batch_weights_real = target * self.real_weight
        batch_weights_fake = (1 - target) * self.fake_weight
        batch_weights = batch_weights_real + batch_weights_fake
        # unpack predictions
        preds, in_mtrx, feat_mtrx = preds
        ce_loss_val = self.criterion(preds, target, weight=batch_weights)
        reg_loss = 0
        if in_mtrx is not None:
            reg_loss += self._regularization_term(in_mtrx)
        if feat_mtrx is not None:
            reg_loss += self._regularization_term(feat_mtrx)
        reg_loss = self.alpha * reg_loss / float(preds.size(0))
        return ce_loss_val + reg_loss