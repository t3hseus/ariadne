import gin
import torch
from pytorch_lightning.metrics import functional as metrics

from ariadne.point_net.model import MIN_EPS_HOLDER

# TODO: update all metrics!!!


@gin.configurable('point_net.precision', allowlist=[])
def precision(preds, target):
    return metrics.precision(
        preds > 0.5, target, class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net.recall', allowlist=[])
def recall(preds, target):
    return metrics.recall(
        preds > 0.5, target, class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net.f1_score', allowlist=[])
def f1_score(preds, target):
    return metrics.f1((preds > 0.5).int(), target.int(), num_classes=2)


@gin.configurable('point_net.accuracy', allowlist=[])
def accuracy(preds, target):
    return metrics.accuracy(preds > 0.5, target, num_classes=2)





#v2=========

@gin.configurable('point_net_v2.precision', allowlist=[])
def precision_v2(all_preds, target):
    preds, _, _ = all_preds
    return metrics.precision((preds > 0.5), target.int(), is_multiclass=False)


@gin.configurable('point_net_v2.recall', allowlist=[])
def recall_v2(all_preds, target):
    preds, _, _ = all_preds
    return metrics.recall((preds > 0.5), target.int(), is_multiclass=False)


@gin.configurable('point_net_v2.f1_score', allowlist=[])
def f1_score_v2(all_preds, target):
    preds, _, _ = all_preds
    return metrics.f1((preds > 0.5).int(), target.int(), num_classes=2, multilabel=False)


@gin.configurable('point_net_v2.accuracy', allowlist=[])
def accuracy_v2(all_preds, target):
    preds, _, _ = all_preds
    return metrics.accuracy((preds > 0.5), target.int())

#v2=========



@gin.configurable('point_net_impulse.precision', allowlist=[])
def precision(preds, target):
    return metrics.precision(
        preds[:, :, 0] > 0.5, target[:, :, 0], class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net_impulse.recall', allowlist=[])
def recall(preds, target):
    return metrics.recall(
        preds[:, :, 0] > 0.5, target[:, :, 0], class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net_impulse.f1_score', allowlist=[])
def f1_score(preds, target):
    return metrics.f1(
        (preds[:, :, 0] > 0.5).int(),
        (target[:, :, 0]).int(),
        num_classes=2)


@gin.configurable('point_net_impulse.accuracy', allowlist=[])
def accuracy(preds, target):
    return metrics.accuracy(preds[:, :, 0] > 0.5, target[:, :, 0], num_classes=2)


@gin.configurable('point_net_impulse.x_diff', allowlist=[])
def x_diff(preds, target):
    return torch.norm(preds[:, :, 1] - target[:, :, 1])


@gin.configurable('point_net_impulse.y_diff', allowlist=[])
def y_diff(preds, target):
    return torch.norm(preds[:, :, 2] - target[:, :, 2])


@gin.configurable('point_net_impulse.z_diff', allowlist=[])
def z_diff(preds, target):
    return torch.norm(preds[:, :, 3] - target[:, :, 3])


@gin.configurable('point_net_impulse.norm', allowlist=[])
def norm(preds, target):
    return torch.norm(preds[:, :, 1:] - target[:, :, 1:])


@gin.configurable('point_net_impulse.norm_for_true', allowlist=[])
def norm_true(preds, target):
    tgt_true = target[target[:, :, 0] > 0.5]

    preds_true = preds[target[:, :, 0] > 0.5]
    return torch.norm(preds_true[:, 1:] - tgt_true[:, 1:])

@gin.configurable('point_net_dist.recall_dist_for_true', allowlist=[])
def recall_dist_for_true(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_and(preds_true, tgt_true)

    return torch.norm(preds[only_true] - target[only_true])


@gin.configurable('point_net_dist.precision_dist_for_true', allowlist=[])
def precision_dist_for_true(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_or(preds_true, tgt_true)

    return torch.norm(preds[only_true] - target[only_true])


@gin.configurable('point_net_dist.wrong_dist', allowlist=[])
def wrong_dist(preds, target):
    preds_true = (preds < MIN_EPS_HOLDER.MIN_EPS)

    tgt_true = (target < MIN_EPS_HOLDER.MIN_EPS)

    only_true = torch.logical_xor(preds_true, tgt_true)

    # count_per_batch = torch.floor_divide(only_true.sum(dim=1), only_true.shape[1])

    return only_true.double().mean()


@gin.configurable('point_net_dist.norm', allowlist=[])
def norm(preds, target):
    return torch.norm(preds - target)


@gin.configurable('point_net_dist.count_in_eps', allowlist=[])
def count_in_eps(preds, target):
    return torch.div((target < MIN_EPS_HOLDER.MIN_EPS).sum(), (target == 0).sum().double())


@gin.configurable('point_net_dist.close_points_dist_exp', allowlist=[])
def close_points_dist_exp(preds, target):
    batch_tgt_inv = 1 - target + MIN_EPS_HOLDER.MIN_EPS
    # we leave only very close points
    batch_tgt_inv = torch.exp(batch_tgt_inv)

    batch_pred_inv = torch.clamp(1 - preds, 0, 1) + MIN_EPS_HOLDER.MIN_EPS
    batch_pred_inv = torch.exp(batch_pred_inv)

    return torch.norm(batch_pred_inv - batch_tgt_inv)


@gin.configurable('point_net_dist.close_points_dist_div', allowlist=[])
def close_points_dist_div(preds, target):
    batch_tgt_inv = 1 - target + MIN_EPS_HOLDER.MIN_EPS
    # we leave only very close points
    batch_tgt_inv = 1. / batch_tgt_inv

    batch_pred_inv = torch.clamp(1 - preds, 1e-5, 1) + MIN_EPS_HOLDER.MIN_EPS
    batch_pred_inv = 1. / batch_pred_inv

    return torch.norm(batch_pred_inv - batch_tgt_inv)


@gin.configurable('point_net_dist.precision', allowlist=[])
def precision_dist(preds, target):
    return metrics.precision(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).int(), (target == 0.0).int(), class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net_dist.recall', allowlist=[])
def recall_dist(preds, target):
    return metrics.recall(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).int(), (target == 0.0).int(), class_reduction='none', num_classes=2)[1]


@gin.configurable('point_net_dist.f1_score', allowlist=[])
def f1_score_dist(preds, target):
    return metrics.f1(
        (preds < MIN_EPS_HOLDER.MIN_EPS / 2).int(), (target == 0.0).int(), num_classes=2)[1]


@gin.configurable('point_net_dist.accuracy', allowlist=[])
def accuracy_dist(preds, target):
    return metrics.accuracy((preds < MIN_EPS_HOLDER.MIN_EPS / 2).int(), (target == 0.0).int(), num_classes=2)
