import gin
import torch
from pytorch_lightning.metrics import functional as metrics
from pytorch_lightning.metrics import Metric


def _compute_metric(preds, target, is_softmax, is_logits, metric_func, unpack_method, **kwargs):
    target = target.int()
    if unpack_method is not None:
        preds = unpack_method(preds)
    if is_softmax:
        # (N, Classes, ...)
        preds = torch.argmax(preds, dim=1).int()
    else:
        if is_logits:
            preds = torch.sigmoid(preds)
        preds = (preds > 0.5).int()
    return metric_func(preds, target, **kwargs)


@gin.configurable('precision', allowlist=['is_softmax', 'is_logits', 'unpack_method'])
def precision(preds, target, is_softmax=False, is_logits=False, unpack_method=None):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        is_logits=is_logits,
        metric_func=metrics.precision,
        is_multiclass=False,
        unpack_method=unpack_method
    )

@gin.configurable('recall', allowlist=['is_softmax', 'is_logits', 'unpack_method'])
def recall(preds, target, is_softmax=False, is_logits=False, unpack_method=None):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        is_logits=is_logits,
        metric_func=metrics.recall,
        is_multiclass=False,
        unpack_method=unpack_method
    )

@gin.configurable('f1_score', allowlist=['is_softmax', 'is_logits', 'unpack_method'])
def f1_score(preds, target, is_softmax=False, is_logits=False, unpack_method=None):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        is_logits=is_logits,
        metric_func=metrics.f1,
        num_classes=2,
        average='none',
        unpack_method=unpack_method
    )[1]

@gin.configurable('accuracy', allowlist=['is_softmax', 'is_logits', 'unpack_method'])
def accuracy(preds, target, is_softmax=False, is_logits=False, unpack_method=None):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        is_logits=is_logits,
        metric_func=metrics.accuracy,
        unpack_method=unpack_method
    )