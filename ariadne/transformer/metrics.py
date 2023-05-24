import gin
import torch
from torchmetrics import functional as metrics
import torch.nn.functional as f


@gin.configurable('transformer.precision', allowlist=[])
def precision(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)
    return metrics.precision(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary'
    )


@gin.configurable('transformer.recall', allowlist=[])
def recall(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)
    return metrics.recall(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary'
    )


@gin.configurable('transformer.f1_score', allowlist=[])
def f1_score(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)

    return metrics.f1_score(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary')


@gin.configurable('transformer.accuracy', allowlist=[])
def accuracy(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)
    return metrics.accuracy(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary'
    )



@gin.configurable('transformer.fake_acc', allowlist=[])
def fake_accuracy(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)
    mask *= (target['y'].reshape(-1) == 0)
    if not mask.sum():
        return -1.
    return metrics.accuracy(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary'
    )


@gin.configurable('transformer.true_acc', allowlist=[])
def true_accuracy(preds, target):
    mask = target['mask'].to(torch.bool).reshape(-1)
    mask *= (target['y'].reshape(-1) == 1)
    if not mask.sum():
        return -1.
    return metrics.accuracy(
        f.sigmoid(preds.reshape(-1).masked_select(mask)),
        target['y'].reshape(-1).masked_select(mask).to(int),
        average='micro',
        task='binary'
    )
