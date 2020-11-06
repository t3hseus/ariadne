import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

gin.external_configurable(CrossEntropyLoss)




