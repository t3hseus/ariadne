import gin
from pytorch_lightning.metrics import functional as metrics


@gin.configurable('graph_net.precision', allowlist=[])
def precision(preds, target):
    return metrics.precision(
        preds > 0.5, target, class_reduction='none', num_classes=2)[1]

@gin.configurable('graph_net.recall', allowlist=[])
def recall(preds, target):
    return metrics.recall(
        preds > 0.5, target, class_reduction='none', num_classes=2)[1]

@gin.configurable('graph_net.f1_score', allowlist=[])
def f1_score(preds, target):
    return metrics.f1((preds > 0.5).int(), target.int(), num_classes=2)

@gin.configurable('graph_net.accuracy', allowlist=[])
def accuracy(preds, target):
    return metrics.accuracy(preds > 0.5, target, num_classes=2)
