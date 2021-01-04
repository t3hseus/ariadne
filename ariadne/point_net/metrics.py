import gin
from pytorch_lightning.metrics import functional as metrics


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
