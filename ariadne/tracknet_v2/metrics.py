import math
import gin
import torch
import numpy as np
import torch.nn.functional as F


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
def point_in_ellipse(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    # unpack target = (target, mask)
    target, mask = target
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

    x_dist = (preds[:, :, 0] - target[:, :, 0]) ** 2
    y_dist = (preds[:, :, 1] - target[:, :, 1]) ** 2
    x_part = x_dist / torch.pow(preds[:, :, 2], 2)
    y_part = y_dist / torch.pow(preds[:, :, 3], 2)
    # left size of equation x_part + y_part = 1
    left_side = x_part + y_part
    result = left_side <= 1
    if mask is not None:
        result_shape = result.shape
        return result.masked_select(mask).view(len(result), -1)
    return result

@gin.configurable(allowlist=[])
def efficiency(preds, target):
    """Checks if the next point of track segment
    is located in the predicted circle
    1 - if yes, 0 - otherwise
    """
    idx = point_in_ellipse(preds, target)
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
    for i in range(2, tracklen):
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
