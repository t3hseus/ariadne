import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointInEllipseLoss(nn.Module):
    """Penalizes predictions when a true point
    is far from the center of the predicted ellipse

    # Arguments
        preds: output of the model (x, y, r1, r2)
        targets: loss targets (x_true, y_true)
    """
    def __init__(self):
        super(PointInEllipseLoss, self).__init__()

    def forward(self, preds, target):
        if preds.size(0) != target.size(0):
            raise ValueError('Shape mismatch! Number of samples in '
                            'the prediction and target must be equal. '
                            f'{preds.size(0) != target.size(0)}')

        if preds.size(1) != 4:
            raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2), '
                             f'but got preds.shape[1] = {preds.size(1)}')

        if target.size(1) != 2:
            raise ValueError('Target must be 2-dimensional (x, y), '
                             f'but got target.shape[1] = {target.size(1)}')

        x_coord_loss = (preds[:, 0] - target[:, 0]) / preds[:, 2]
        y_coord_loss = (preds[:, 1] - target[:, 1]) / preds[:, 3]
        squared_loss = torch.pow(x_coord_loss, 2) + torch.pow(y_coord_loss, 2)
        return torch.sqrt(squared_loss)


class EllipseSquareLoss(nn.Module):
    """Penalizes ellipses with a big square
    PI value is ommited from square formula

    # Arguments
        preds: output of the model (x, y, r1, r2)
    """
    def __init__(self):
        super(EllipseSquareLoss, self).__init__()

    def forward(self, preds):
        if preds.size(1) != 4:
            raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2), '
                             f'but got preds.shape[1] = {preds.size(1)}')

        return preds[:, 2] * preds[:, 3]


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
        points_in_ellipse = self.point_in_ellipse_loss(preds, target)
        ellipses_square = self.ellipse_square_loss(preds)
        loss = self.alpha * points_in_ellipse + (1 - self.alpha) * ellipses_square
        return torch.mean(loss).float()
