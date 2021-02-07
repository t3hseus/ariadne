import gin
import torch
from pytorch_lightning.metrics import functional as metrics

from ariadne.point_net.model import MIN_EPS_HOLDER


@gin.configurable('point_net.precision', whitelist=[])
def precision(preds, target):
    return metrics.precision(
        preds > 0.5, target, reduction='none', num_classes=2)[1]


@gin.configurable('point_net.recall', whitelist=[])
def recall(preds, target):
    return metrics.recall(
        preds > 0.5, target, reduction='none', num_classes=2)[1]


@gin.configurable('point_net.f1_score', whitelist=[])
def f1_score(preds, target):
    return metrics.f1_score(
        preds > 0.5, target, reduction='none', num_classes=2)[1]


@gin.configurable('point_net.accuracy', whitelist=[])
def accuracy(preds, target):
    return metrics.accuracy(preds > 0.5, target, num_classes=2)


@gin.configurable('point_net_impulse.precision', whitelist=[])
def precision(preds, target):
    return metrics.precision(
        preds[:, :, 0] > 0.5, target[:, :, 0], reduction='none', num_classes=2)[1]


@gin.configurable('point_net_impulse.recall', whitelist=[])
def recall(preds, target):
    return metrics.recall(
        preds[:, :, 0] > 0.5, target[:, :, 0], reduction='none', num_classes=2)[1]


@gin.configurable('point_net_impulse.f1_score', whitelist=[])
def f1_score(preds, target):
    return metrics.f1_score(
        preds[:, :, 0] > 0.5, target[:, :, 0], reduction='none', num_classes=2)[1]


@gin.configurable('point_net_impulse.accuracy', whitelist=[])
def accuracy(preds, target):
    return metrics.accuracy(preds[:, :, 0] > 0.5, target[:, :, 0], num_classes=2)


@gin.configurable('point_net_impulse.x_diff', whitelist=[])
def x_diff(preds, target):
    return torch.norm(preds[:, :, 1] - target[:, :, 1])


@gin.configurable('point_net_impulse.y_diff', whitelist=[])
def y_diff(preds, target):
    return torch.norm(preds[:, :, 2] - target[:, :, 2])


@gin.configurable('point_net_impulse.z_diff', whitelist=[])
def z_diff(preds, target):
    return torch.norm(preds[:, :, 3] - target[:, :, 3])


@gin.configurable('point_net_impulse.norm', whitelist=[])
def norm(preds, target):
    return torch.norm(preds[:, :, 1:] - target[:, :, 1:])


@gin.configurable('point_net_impulse.norm_for_true', whitelist=[])
def norm_true(preds, target):
    tgt_true = target[target[:, :, 0] > 0.5]

    preds_true = preds[target[:, :, 0] > 0.5]
    return torch.norm(preds_true[:, 1:] - tgt_true[:, 1:])

@gin.configurable('point_net_dist.recall_dist_for_true', whitelist=[])
def recall_dist_for_true(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_and(preds_true, tgt_true)

    return torch.norm(preds[only_true] - target[only_true])


@gin.configurable('point_net_dist.precision_dist_for_true', whitelist=[])
def precision_dist_for_true(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_or(preds_true, tgt_true)

    return torch.norm(preds[only_true] - target[only_true])


@gin.configurable('point_net_dist.wrong_dist', whitelist=[])
def wrong_dist(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_xor(preds_true, tgt_true)

    # count_per_batch = torch.floor_divide(only_true.sum(dim=1), only_true.shape[1])

    return only_true.double().mean()


@gin.configurable('point_net_dist.norm', whitelist=[])
def norm(preds, target):
    return torch.norm(preds - target)


@gin.configurable('point_net_dist.count_in_eps', whitelist=[])
def count_in_eps(preds, target):
    return torch.div((target < MIN_EPS_HOLDER.MIN_EPS).sum(), (target == 0).sum().double())


@gin.configurable('point_net_dist.close_points_dist_exp', whitelist=[])
def close_points_dist_exp(preds, target):
    batch_tgt_inv = 1 - target + MIN_EPS_HOLDER.MIN_EPS
    # we leave only very close points
    batch_tgt_inv = torch.exp(batch_tgt_inv)

    batch_pred_inv = torch.clamp(1 - preds, 0, 1) + MIN_EPS_HOLDER.MIN_EPS
    batch_pred_inv = torch.exp(batch_pred_inv)

    return torch.norm(batch_pred_inv - batch_tgt_inv)


@gin.configurable('point_net_dist.close_points_dist_div', whitelist=[])
def close_points_dist_div(preds, target):
    batch_tgt_inv = 1 - target + MIN_EPS_HOLDER.MIN_EPS
    # we leave only very close points
    batch_tgt_inv = 1. / batch_tgt_inv

    batch_pred_inv = torch.clamp(1 - preds, 1e-5, 1) + MIN_EPS_HOLDER.MIN_EPS
    batch_pred_inv = 1. / batch_pred_inv

    return torch.norm(batch_pred_inv - batch_tgt_inv)


@gin.configurable('point_net_dist.precision', whitelist=[])
def precision_dist(preds, target):
    return metrics.precision(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).long(), (target == 0.0).long(), reduction='none', num_classes=2)[1]


@gin.configurable('point_net_dist.recall', whitelist=[])
def recall_dist(preds, target):
    return metrics.recall(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).long(), (target == 0.0).long(), reduction='none', num_classes=2)[1]


@gin.configurable('point_net_dist.f1_score', whitelist=[])
def f1_score_dist(preds, target):
    return metrics.f1_score(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).long(), (target == 0.0).long(), reduction='none', num_classes=2)[1]


@gin.configurable('point_net_dist.accuracy', whitelist=[])
def accuracy_dist(preds, target):
    return metrics.accuracy((preds < MIN_EPS_HOLDER.MIN_EPS / 2).long(), (target == 0.0).long(), num_classes=2)
