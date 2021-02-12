import math
import gin
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import LogSoftmax,Softmax

import gin
from pytorch_lightning.metrics import functional as metrics


@gin.configurable('tracknet_v2_1.precision', allowlist=[])
def precision(preds, target):
    softmax = Softmax()
    preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.precision(
        (preds > 0.5).int(), target, reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.recall', allowlist=[])
def recall(preds, target):
    softmax = Softmax()
    preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.recall(
        (preds > 0.5).int(), target.int(), reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.f1_score', allowlist=[])
def f1_score(preds, target):
    softmax = Softmax()
    changed_preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.f1_score(
        (preds > 0.5).int(), target.int(), num_classes=2)

@gin.configurable('tracknet_v2_1.accuracy', allowlist=[])
def accuracy(preds, target):
    return metrics.accuracy((preds > 0.5).int(), target.int(), num_classes=2)


