import math
import gin
import torch
import numpy as np

@gin.configurable(allowlist=[])
def ellipse_area(preds, target):
    """Computes the area of predicted ellipses

    # Arguments
        preds: output of the model (x, y, r1, r2)
    """
    # unpack target = (target, mask)
    target, mask = target
    if preds.size(2) != 4:
        raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2), '
                         f'but got preds.shape[2] = {preds.size(2)}')
    areas = preds[:, :, 2] * preds[:, :, 3] * math.pi
    if mask is not None:
        return areas.masked_select(mask).mean().float()
    return areas.mean().float()

@gin.configurable(allowlist=[])
def ellipse_area_xyz(preds, target):
    """Computes the area of predicted ellipses

    # Arguments
        preds: output of the model (x, y, z, r1, r2, r3)
    """
    # unpack target = (target, mask)
    target, mask = target
    if preds.size(2) != 6:
        raise ValueError('Prediction must be 6-dimensional (x, y, z, r1, r2, r3), '
                         f'but got preds.shape[2] = {preds.size(2)}')
    areas = preds[:, :, 3] * preds[:, :, 4] * preds[:, :, 5] * math.pi
    if mask is not None:
        return areas.masked_select(mask).mean().float()
    return areas.mean().float()


@gin.configurable(allowlist=[])
def point_in_ellipse(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    if preds.shape[0] != target.shape[0]:
        raise ValueError('Shape mismatch! Number of samples in '
                        'the prediction and target must be equal. '
                        f'{preds.size(0) != target.size(0)}')

    if (preds.shape[-1] < 4):
        raise ValueError('Prediction must be 4-dimensional (x, y, r1, r2) or 5-dimensional (x, y, z, r1, r2), '
                            f'but got preds.shape[2] = {preds.size(2)}')

    if target.shape[-1] < 2:
        raise ValueError('Target must be 2-dimensional (x, y) or 3-dimensional (x, y, z), '
                             f'but got target.shape[2] = {target.size(2)}')
    if preds.ndim < 3 and target.ndim == 3:
        preds = preds.unsqueeze(1).repeat(1, target.shape[1], 1)

    x_dist = (preds[:, :, 0] - target[:, :, 0]) ** 2
    y_dist = (preds[:, :, 1] - target[:, :, 1]) ** 2
    x_part = x_dist / torch.pow(preds[:, :, -2], 2)
    y_part = y_dist / torch.pow(preds[:, :, -1], 2)
    # left size of equation x_part + y_part = 1
    left_side = x_part + y_part
    if target.size(2) == 3:
        is_this_station = torch.eq(preds[:, :, 2], target[:, :, 2])
        return torch.le(left_side, 1) * is_this_station
    return torch.le(left_side, 1)


@gin.configurable(allowlist=[])
def point_in_ellipse_xyz(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    if preds.shape[0] != target.shape[0]:
        raise ValueError('Shape mismatch! Number of samples in '
                        'the prediction and target must be equal. '
                        f'{preds.size(0) != target.size(0)}')

    if (preds.shape[-1] < 6):
        raise ValueError('Prediction must be 6-dimensional (x, y, z, r1, r2, r3), '
                         f'but got preds.shape[2] = {preds.size(2)}')

    if target.shape[-1] < 3:
        raise ValueError('Target must be 3-dimensional (x, y, z), '
                             f'but got target.shape[2] = {target.size(2)}')
    if preds.ndim < 3 and target.ndim == 3:
        preds = preds.unsqueeze(1).repeat(1, target.shape[1], 1)

    x_dist = (preds[:, :, 0] - target[:, :, 0]) ** 2
    y_dist = (preds[:, :, 1] - target[:, :, 1]) ** 2
    z_dist = (preds[:, :, 2] - target[:, :, 2]) ** 2
    x_part = x_dist / torch.pow(preds[:, :, 3], 2)
    y_part = y_dist / torch.pow(preds[:, :, 4], 2)
    z_part = z_dist / torch.pow(preds[:, :, 5], 2)
    
    # left size of equation x_part + y_part = 1
    left_side = x_part + y_part + z_part
    #if target.size(2) == 3:
    #    is_this_station = torch.eq(preds[:, :, 2], target[:, :, 2])
    #    return torch.le(left_side, 1) * is_this_station
    return torch.le(left_side, 1)


@gin.configurable(allowlist=[])
def efficiency(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    # unpack target = (target, mask)
    target, mask = target
    idx = point_in_ellipse(preds, target)
    if mask is not None:
        idx = idx.masked_select(mask)
    return torch.sum(idx.float()) / len(idx)

@gin.configurable(allowlist=[])
def efficiency_xyz(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    # unpack target = (target, mask)
    target, mask = target
    idx = point_in_ellipse_xyz(preds, target)
    if mask is not None:
        idx = idx.masked_select(mask)
    return torch.sum(idx.float()) / len(idx)


# TODO: fix this function
@gin.configurable(allowlist=[])
def calc_metrics(inputs, model, tracklen=None):
    """ Take array of tracks in the format of numpy 3d array
    [[[x1, y1, z1, ...], [x2, y2, z2, ...]], [...]],
    array has shape (batch_size, n_hits, n_features).
    Then for each of the tracks apply model to prolong each
    track from the first two hits to the end of the track.

    # Arguments
        inputs: array of tracks
        model: torch model
        tracklen: size of the tracks in a batch

    # Returns
        efficiency: fraction of tracks which model processed without errors
        hits_efficiency: fraction of hits predicted correctly
    """
    if len(inputs.shape) != 3:
        raise ValueError('Input array must be 3-dimensional, '
                         f'but got {inputs.shape}')

    if inputs.shape[1] <= 2:
        raise ValueError(f'Input shape is {inputs.shape}, '
                         'second dimension must be greater than 2')

    if tracklen is not None:
        if tracklen <= 2:
            raise ValueError(f'Invalid tracklen, must be >2, got {tracklen} instead')
        if tracklen > inputs.shape[1]:
            raise ValueError('Tracklen can`t be greater than number of hits in the sample '
                             f'inputs.shape=[{inputs.shape}], tracklen={tracklen}')

    efficiency = 0
    hits_efficiency = 0
    # create tensor and place it to CPU or GPU
    inputs_val = torch.from_numpy(inputs).to(model.device)
    model.eval()

    if tracklen is None:
        tracklen = inputs_val.size(1)

    # run from the first station to the last
    for i in range(1, tracklen):
        # cut off all track-candidates
        inputs_part = inputs_val[:, :i]
        # x, y coords of the next hit in a track
        target = inputs_val[:, i, :2]
        # get model's prediction
        preds = model(inputs_part)

        # get indices of the tracks, which continuation was found
        idx = point_in_ellipse(preds, target)
        # count number of right predictions
        hits_efficiency += np.sum(idx) / len(idx)
        # exclude
        inputs_val = inputs_val[idx]

    # count number of track for which we found all points
    efficiency = len(inputs_val) / len(inputs)
    # recompute the hits_efficiency
    hits_efficiency = (hits_efficiency + 2) / tracklen
    return efficiency, hits_efficiency
