import gin
import torch
from pytorch_lightning.metrics import functional as metrics
from pytorch_lightning.metrics import Metric


def _compute_metric(preds, target, is_softmax, metric_func, **kwargs):
    target = target.int()
    if is_softmax:
        # (N, Classes, ...)
        preds = torch.argmax(preds, dim=1).int()
    else:
        preds = (preds > 0.5).int()
    return metric_func(preds, target, **kwargs)


@gin.configurable('precision', allowlist=['is_softmax'])
def precision(preds, target, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.precision,
        is_multiclass=False
    )

@gin.configurable('recall', allowlist=['is_softmax'])
def recall(preds, target, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.recall,
        is_multiclass=False
    )

@gin.configurable('f1_score', allowlist=['is_softmax'])
def f1_score(preds, target, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.f1,
        num_classes=2,
        average='none'
    )[1]

@gin.configurable('accuracy', allowlist=['is_softmax'])
def accuracy(preds, target, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.accuracy
    )