import gin
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, feature_transform=False, n_feat=3):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(n_feat, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1)


@gin.configurable
class PointNetSeg_v1(nn.Module):
    def __init__(self, feature_transform=False, n_feat=3):
        super(PointNetSeg_v1, self).__init__()
        MIN_EPS_HOLDER()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(feature_transform=feature_transform, n_feat=n_feat)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return F.sigmoid(x).squeeze(-1)


@gin.configurable('point_net_dist.MIN_EPS_HOLDER')
def MIN_EPS_HOLDER(MIN_EPS):
    return MIN_EPS_HOLDER.__setattr__('MIN_EPS', MIN_EPS)

