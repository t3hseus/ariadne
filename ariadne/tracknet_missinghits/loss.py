import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointInEllipseLossMissingHits(nn.Module):
    def __init__(self):
        super(PointInEllipseLossMissingHits, self).__init__()

    def forward(self, preds, target):
        if preds.size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                            'the prediction and target must be equal. '
                            f'{preds.size(0) != target.size(0)}')

        if (preds.shape[-1] < 8):
            raise ValueError('Prediction must be 8-dimensional (x1, y1, z1, r1, x2, y2, z2, r2), '
                             f'but got preds.shape[2] = {preds.size(2)}')

        if target.shape[-1] < 3:
            raise ValueError('Target must be 3-dimensional (x, y, z), '
                            f'but got target.shape[2] = {target.size(2)}')
            
        #preds = preds#[:-1]

        target_first = target[:, :-1]
        x1_coord_loss = (preds[:, :, 0] - target_first[:, :, 0]) / preds[:, :, 3]
        y1_coord_loss = (preds[:, :, 1] - target_first[:, :, 1]) / preds[:, :, 3]
        z1_coord_loss = (preds[:, :, 2] - target_first[:, :, 2]) / preds[:, :, 3]
        squared_loss1 = torch.pow(x1_coord_loss, 2) + torch.pow(y1_coord_loss, 2) + torch.pow(z1_coord_loss, 2)
        
        target_second = target[:, 1:]
        x2_coord_loss = (preds[:, :, 4] - target_second[:, :, 0]) / preds[:, :, 7]
        y2_coord_loss = (preds[:, :, 5] - target_second[:, :, 1]) / preds[:, :, 7]
        z2_coord_loss = (preds[:, :, 6] - target_second[:, :, 2]) / preds[:, :, 7]
        squared_loss2 = torch.pow(x2_coord_loss, 2) + torch.pow(y2_coord_loss, 2) + torch.pow(z2_coord_loss, 2)
        
        squared_loss = torch.cat((squared_loss1, squared_loss2), dim=1)
        return torch.sqrt(squared_loss)

class EllipseSquareLossMissingHits(nn.Module):
    def __init__(self):
        super(EllipseSquareLossMissingHits, self).__init__()

    def forward(self, preds):
        if (preds.shape[-1] < 8):
            raise ValueError('Prediction must be 8-dimensional (x1, y1, z1, r1, x2, y2, z2, r2), '
                             f'but got preds.shape[2] = {preds.size(2)}')
        square_1 = preds[:, :, 3] ** 2
        square_2 = preds[:, :, 7] ** 2
        loss = torch.cat((square_1, square_2), dim=1)
        return loss


@gin.configurable
class TrackNetLossMissingHits(nn.Module):
    def __init__(self, alpha=0.9):
        super(TrackNetLossMissingHits, self).__init__()

        if alpha > 1 or alpha < 0:
            raise ValueError('Weighting factor alpha must be in range [0, 1], '
                             f'but got alpha={alpha}')
        self.alpha = alpha
        self.point_in_ellipse_loss = PointInEllipseLossMissingHits()
        self.ellipse_square_loss = EllipseSquareLossMissingHits()

    def forward(self, preds, target):
        # unpack target = (target, mask)
        target, mask = target
        points_in_ellipse = self.point_in_ellipse_loss(preds, target)
        ellipses_square = self.ellipse_square_loss(preds)
        loss = self.alpha * points_in_ellipse + (1 - self.alpha) * ellipses_square
        if mask is not None:
            return loss.masked_select(mask).mean().float()
        return loss.mean().float()
