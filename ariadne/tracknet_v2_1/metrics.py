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
        preds, target, class_reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.recall', allowlist=[])
def recall(preds, target):
    softmax = Softmax()
    preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.recall(
        preds, target, class_reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.f1_score', allowlist=[])
def f1_score(preds, target):
    softmax = Softmax()
    #print(preds)
    changed_preds = torch.argmax(softmax(preds.float()), dim=1)
    #print(target)
    #print(changed_preds)
    return metrics.f1_score(
        preds, target, class_reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.accuracy', allowlist=[])
def accuracy(preds, target):
    softmax = Softmax()
    changed_preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.accuracy(changed_preds, target, num_classes=2)