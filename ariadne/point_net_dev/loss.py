import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


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
