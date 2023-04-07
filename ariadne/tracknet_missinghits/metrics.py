import math
import gin
import torch
import numpy as np


@gin.configurable(allowlist=[])
def ellipse_area_missinghits(preds, target):
    """Computes the area of predicted ellipses

    # Arguments
        preds: output of the model (x, y, z, r)
    """
    # unpack target = (target, mask)
    target, mask = target
    if preds.size(2) != 8:
        raise ValueError('Prediction must be 8-dimensional (x1, y1, z1, r1, x2, y2, z2, r2), '
                         f'but got preds.shape[2] = {preds.size(2)}')
    radius_1 = preds[:, :, 3]
    radius_2 = preds[:, :, 7]
    
    areas_1 = (4./3.) * math.pi * radius_1**3
    areas_2 = (4./3.) * math.pi * radius_2**3
    areas = torch.cat((areas_1, areas_2), dim=1)
    
    if mask is not None:
        return areas.masked_select(mask).mean().float()
    return areas.mean().float()


@gin.configurable(allowlist=[])
def point_in_ellipse_missinghits(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    if preds.shape[0] != target.shape[0]:
        raise ValueError('Shape mismatch! Number of samples in '
                        'the prediction and target must be equal. '
                        f'{preds.size(0) != target.size(0)}')

    if (preds.shape[-1] < 8):
        raise ValueError('Prediction must be 8-dimensional (x1, y1, z1, r1, x2, y2, z2, r2), '
                         f'but got preds.shape[2] = {preds.size(2)}')

    if target.shape[-1] < 3:
        raise ValueError('Target must be 3-dimensional (x, y, z), '
                             f'but got target.shape[2] = {target.size(2)}')
    if preds.ndim < 3 and target.ndim == 3:
        preds = preds.unsqueeze(1).repeat(1, target.shape[1], 1)

    target_first = target[:, :-1]
    x1_dist = (preds[:, :, 0] - target_first[:, :, 0]) ** 2
    y1_dist = (preds[:, :, 1] - target_first[:, :, 1]) ** 2
    z1_dist = (preds[:, :, 2] - target_first[:, :, 2]) ** 2
    x1_part = x1_dist / torch.pow(preds[:, :, 3], 2)
    y1_part = y1_dist / torch.pow(preds[:, :, 3], 2)
    z1_part = z1_dist / torch.pow(preds[:, :, 3], 2)
    
    target_second = target[:, 1:]
    x2_dist = (preds[:, :, 4] - target_second[:, :, 0]) ** 2
    y2_dist = (preds[:, :, 5] - target_second[:, :, 1]) ** 2
    z2_dist = (preds[:, :, 6] - target_second[:, :, 2]) ** 2
    x2_part = x2_dist / torch.pow(preds[:, :, 7], 2)
    y2_part = y2_dist / torch.pow(preds[:, :, 7], 2)
    z2_part = z2_dist / torch.pow(preds[:, :, 7], 2)
    
    left_side1 = x1_part + y1_part + z1_part
    left_side2 = x2_part + y2_part + z2_part
    
    left_side = torch.cat((left_side1, left_side2), dim=1)
    return torch.le(left_side, 1)


@gin.configurable(allowlist=[])
def efficiency_missinghits(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    target, mask = target
    idx = point_in_ellipse_missinghits(preds, target)
    if mask is not None:
        idx = idx.masked_select(mask)
    return idx.float().mean()
