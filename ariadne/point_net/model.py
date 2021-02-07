import gin
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, scale_factor, n_feat=3):
        super(PointNetfeat, self).__init__()
        self.scale_factor = 8 * scale_factor


        # if scale_factor = 2           3           16
        self.conv1 = torch.nn.Conv1d(n_feat, self.scale_factor, 1)
        #                                   16                  32
        self.conv4 = torch.nn.Conv1d(self.scale_factor, self.scale_factor * 2, 1)
        #                                   32                      64
        self.conv2 = torch.nn.Conv1d(self.scale_factor * 2, self.scale_factor * 4, 1)
        #                                   64                  128
        self.conv3 = torch.nn.Conv1d(self.scale_factor * 4, self.scale_factor * 8, 1)
        #                                   128                 64
        self.fc1 = torch.nn.Linear(self.scale_factor * 8, self.scale_factor * 4)
        #                                   64                  32
        self.fc2 = torch.nn.Linear(self.scale_factor * 4, self.scale_factor * 2)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        self.bn1 = nn.BatchNorm1d(self.scale_factor)
        self.bn12 = nn.BatchNorm1d(self.scale_factor * 2)

        self.bn2 = nn.BatchNorm1d(self.scale_factor * 4)
        self.bn3 = nn.BatchNorm1d(self.scale_factor * 8)

        self.bn4 = nn.BatchNorm1d(self.scale_factor * 4)
        self.bn5 = nn.BatchNorm1d(self.scale_factor * 2)

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn12(self.conv4(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # max = torch.max(x, 2, keepdim=True)[0]
        # min = torch.min(x, 2, keepdim=True)[0]
        # avg = torch.mean(x, 2, keepdim=True)[0]

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.scale_factor * 8)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = x.view(-1, self.scale_factor * 2, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1)


@gin.configurable
class PointNetSeg_v1(nn.Module):
    def __init__(self, scale_factor, n_feat=3):
        super(PointNetSeg_v1, self).__init__()
        MIN_EPS_HOLDER()
        self.feat = PointNetfeat(scale_factor, n_feat=n_feat)
        self.scale_factor = 32 * scale_factor
        # if scale_factor = 2:          64                  32
        self.conv1 = torch.nn.Conv1d(self.scale_factor, self.scale_factor // 2, 1)
        #                               32                      16
        self.conv2 = torch.nn.Conv1d(self.scale_factor // 2, self.scale_factor // 4, 1)
        #                               16                   1
        self.conv3 = torch.nn.Conv1d(self.scale_factor // 4, 1, 1)

        self.bn1 = nn.BatchNorm1d(self.scale_factor // 2)
        self.bn2 = nn.BatchNorm1d(self.scale_factor // 4)

    def forward(self, x):
        x = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        return F.sigmoid(x).squeeze(-1)


@gin.configurable
class PointNetSegImpulse_v1(nn.Module):
    def __init__(self, scale_factor, n_feat=3):
        super(PointNetSegImpulse_v1, self).__init__()
        MIN_EPS_HOLDER()
        self.feat = PointNetfeat(scale_factor, n_feat=n_feat)
        self.scale_factor = 32 * scale_factor
        self.conv1 = torch.nn.Conv1d(self.scale_factor, self.scale_factor // 2, 1)
        self.conv2 = torch.nn.Conv1d(self.scale_factor // 2, self.scale_factor // 4, 1)
        self.conv3 = torch.nn.Conv1d(self.scale_factor // 4, 4, 1)

        self.bn1 = nn.BatchNorm1d(self.scale_factor // 2)
        self.bn2 = nn.BatchNorm1d(self.scale_factor // 4)

    def forward(self, x):
        x = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        return F.sigmoid(x).squeeze(-1)


@gin.configurable('point_net_dist.MIN_EPS_HOLDER')
def MIN_EPS_HOLDER(MIN_EPS):
    return MIN_EPS_HOLDER.__setattr__('MIN_EPS', MIN_EPS)
