import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointInEllipseLoss(nn.Module):
    """Penalizes predictions when a true point
    is far from the center of the predicted ellipse

    # Arguments
        preds: output of the model (x, y, r1, r2) with shape [batch, timesteps, 4]
        targets: loss targets (x_true, y_true) with shape [batch, timesteps, 2]
    """
    def __init__(self):
        super(PointInEllipseLoss, self).__init__()

    def forward(self, preds, target):
        if preds.size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                            'the prediction and target must be equal. '
                            f'{preds.size(0) != target.size(0)}')

        if preds.size(2) != 4:
            raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2), '
                             f'but got preds.shape[2] = {preds.size(2)}')

        if target.size(2) != 2:
            raise ValueError('Target must be 2-dimensional (x, y), '
                             f'but got target.shape[2] = {target.size(2)}')

        x_coord_loss = (preds[:, :, 0] - target[:, :, 0]) / preds[:, :, 2]
        y_coord_loss = (preds[:, :, 1] - target[:, :, 1]) / preds[:, :, 3]
        squared_loss = torch.pow(x_coord_loss, 2) + torch.pow(y_coord_loss, 2)
        return torch.sqrt(squared_loss)
    

class PointInEllipseLossXYZ(nn.Module):
    """Penalizes predictions when a true point
    is far from the center of the predicted ellipse

    # Arguments
        preds: output of the model (x, y, z, r1, r2, r3) with shape [batch, timesteps, 6]
        targets: loss targets (x_true, y_true, z_true) with shape [batch, timesteps, 3]
    """
    def __init__(self):
        super(PointInEllipseLossXYZ, self).__init__()

    def forward(self, preds, target):
        if preds.size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                            'the prediction and target must be equal. '
                            f'{preds.size(0) != target.size(0)}')
            
        if (preds.shape[-1] < 6):
            raise ValueError('Prediction must be 6-dimensional (x, y, z, r1, r2, r3), '
                             f'but got preds.shape[2] = {preds.size(2)}')

        if target.shape[-1] < 3:
            raise ValueError('Target must be 3-dimensional (x, y, z), '
                            f'but got target.shape[2] = {target.size(2)}')

        x_coord_loss = (preds[:, :, 0] - target[:, :, 0]) / preds[:, :, 3]
        y_coord_loss = (preds[:, :, 1] - target[:, :, 1]) / preds[:, :, 4]
        z_coord_loss = (preds[:, :, 2] - target[:, :, 2]) / preds[:, :, 5]
        squared_loss = torch.pow(x_coord_loss, 2) + torch.pow(y_coord_loss, 2) + torch.pow(z_coord_loss, 2)
        return torch.sqrt(squared_loss)


class EllipseSquareLoss(nn.Module):
    """Penalizes ellipses with a big square
    PI value is ommited from square formula

    # Arguments
        preds: output of the model (x, y, r1, r2)
            with shape [batch, timesteps, 4]
    """
    def __init__(self):
        super(EllipseSquareLoss, self).__init__()

    def forward(self, preds):
        if preds.size(2) != 4:
            raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2), '
                             f'but got preds.shape[1] = {preds.size(2)}')

        return preds[:, :, 2] * preds[:, :, 3]
    
    
class EllipseSquareLossXYZ(nn.Module):
    """Penalizes ellipses with a big square
    PI value is ommited from square formula

    # Arguments
        preds: output of the model (x, y, z, r1, r2, r3)
            with shape [batch, timesteps, 6]
    """
    def __init__(self):
        super(EllipseSquareLossXYZ, self).__init__()

    def forward(self, preds):
        if (preds.shape[-1] < 6):
            raise ValueError('Prediction must be 6-dimensional (x, y, z, r1, r2, r3), '
                             f'but got preds.shape[2] = {preds.size(2)}')

        return preds[:, :, 3] * preds[:, :, 4] * preds[:, :, 5]


@gin.configurable
class TrackNetLoss(nn.Module):
    """ Calculates TrackNETv2 loss:
    point_in_ellipse = sqrt((x - x') / r1)^2 + ((y - y') / r2)^2)
    ellipse_square = r1 * r2
    L = alpha * point_in_ellipse + (1 - alpha) * ellipse_square

    # Parameters
        alpha: float, weighting factor, must be < 1
    """
    def __init__(self, alpha=0.9):
        super(TrackNetLoss, self).__init__()

        if alpha > 1 or alpha < 0:
            raise ValueError('Weighting factor alpha must be in range [0, 1], '
                             f'but got alpha={alpha}')

        self.alpha = alpha
        self.point_in_ellipse_loss = PointInEllipseLoss()
        self.ellipse_square_loss = EllipseSquareLoss()

    def forward(self, preds, target):
        # unpack target = (target, mask)
        target, mask = target
        points_in_ellipse = self.point_in_ellipse_loss(preds, target)
        ellipses_square = self.ellipse_square_loss(preds)
        loss = self.alpha * points_in_ellipse + (1 - self.alpha) * ellipses_square
        if mask is not None:
            return loss.masked_select(mask).mean().float()
        return loss.mean().float()

    
@gin.configurable
class TrackNetLossXYZ(nn.Module):
    """ Calculates TrackNETv2 loss:
    point_in_ellipse = sqrt((x - x') / r1)^2 + ((y - y') / r2)^2)
    ellipse_square = r1 * r2
    L = alpha * point_in_ellipse + (1 - alpha) * ellipse_square

    # Parameters
        alpha: float, weighting factor, must be < 1
    """
    def __init__(self, alpha=0.9):
        super(TrackNetLossXYZ, self).__init__()

        if alpha > 1 or alpha < 0:
            raise ValueError('Weighting factor alpha must be in range [0, 1], '
                             f'but got alpha={alpha}')

        self.alpha = alpha
        self.point_in_ellipse_loss = PointInEllipseLossXYZ()
        self.ellipse_square_loss = EllipseSquareLossXYZ()

    def forward(self, preds, target):
        # unpack target = (target, mask)
        target, mask = target
        points_in_ellipse = self.point_in_ellipse_loss(preds, target)
        ellipses_square = self.ellipse_square_loss(preds)
        loss = self.alpha * points_in_ellipse + (1 - self.alpha) * ellipses_square
        if mask is not None:
            return loss.masked_select(mask).mean().float()
        return loss.mean().float()