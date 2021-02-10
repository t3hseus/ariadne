import math
import gin
import torch 
import numpy as np
import torch.nn.functional as F
from torch.nn import LogSoftmax,Softmax

import gin
from pytorch_lightning.metrics import functional as metrics


@gin.configurable('tracknet_v2_1.precision', whitelist=[])
def precision(preds, target):
    softmax = Softmax()
    preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.precision(
        preds, target, reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.recall', whitelist=[])
def recall(preds, target):
    softmax = Softmax()
    preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.recall(
        preds, target, reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.f1_score', whitelist=[])
def f1_score(preds, target):
    softmax = Softmax()
    #print(preds)
    changed_preds = torch.argmax(softmax(preds.float()), dim=1)
    #print(target)
    #print(changed_preds)
    return metrics.f1_score(
        preds, target, reduction='none', num_classes=2)[1]

@gin.configurable('tracknet_v2_1.accuracy', whitelist=[])
def accuracy(preds, target):
    softmax = Softmax()
    changed_preds = torch.argmax(softmax(preds.float()), dim=1)
    return metrics.accuracy(changed_preds, target, num_classes=2)