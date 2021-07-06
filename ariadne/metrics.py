import gin
import torch
import torch.nn.functional as f
from pytorch_lightning.metrics import functional as metrics
from pytorch_lightning.metrics import Metric

import torchmetrics.functional

def _compute_metric(preds, target, is_softmax, metric_func, activation=False, **kwargs):
    target = target.int()
    if is_softmax:
        # (N, Classes, ...)
        preds = torch.argmax(preds, dim=1).int()
    else:
        if activation:
            preds = f.sigmoid(preds)
        preds = (preds > 0.5).int()
    return metric_func(preds, target, **kwargs)

## DEPRECATED METRICS:
@gin.configurable('precision', allowlist=['is_softmax','activation'])
def precision(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.precision,
        is_multiclass=False,
        activation=activation
    )

@gin.configurable('recall', allowlist=['is_softmax','activation'])
def recall(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.recall,
        activation=activation,
        is_multiclass=False
    )

@gin.configurable('f1_score', allowlist=['is_softmax','activation'])
def f1_score(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=metrics.f1,
        activation=activation,
        num_classes=2,
        average='none'
    )[1]

@gin.configurable('accuracy', allowlist=['is_softmax','activation'])
def accuracy(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        activation=activation,
        metric_func=metrics.accuracy
    )

## NEW METRICS
@gin.configurable('new_precision', allowlist=['is_softmax','activation'])
def precision_new(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=torchmetrics.functional.precision,
        multiclass=False,
        activation=activation
    )

@gin.configurable('new_recall', allowlist=['is_softmax','activation'])
def recall_new(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=torchmetrics.functional.recall,
        activation=activation,
        multiclass=False
    )

@gin.configurable('new_f1_score', allowlist=['is_softmax','activation'])
def f1_score_new(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        metric_func=torchmetrics.functional.f1,
        activation=activation,
        multiclass=False
    )

@gin.configurable('new_accuracy', allowlist=['is_softmax','activation'])
def accuracy_new(preds, target, activation=False, is_softmax=False):
    return _compute_metric(
        preds=preds,
        target=target,
        is_softmax=is_softmax,
        activation=activation,
        metric_func=torchmetrics.functional.accuracy
    )
