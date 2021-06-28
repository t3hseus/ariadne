import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

@gin.configurable
class LootLoss(nn.Module):

    def __init__(self, z_constraints_min, z_constraints_max):
        super(LootLoss, self).__init__()
        self.z_constraints_min = z_constraints_min
        self.z_constraints_max = z_constraints_max

    
    def denorm_coord(self, coord, min_val, max_val):
        return ((coord + 1) * (max_val - min_val) / 2) + min_val


    def forward(self, preds, target):

        input = self.denorm_coord(target, self.z_constraints_min, self.z_constraints_max)
        output = self.denorm_coord(preds, self.z_constraints_min, self.z_constraints_max)
        loss = F.l1_loss(input, output)
        return torch.mean(loss).float()

@gin.configurable
class LootLossL2(nn.Module):

    def __init__(self, z_constraints_min, z_constraints_max):
        super(LootLossL2, self).__init__()
        self.z_constraints_min = z_constraints_min
        self.z_constraints_max = z_constraints_max

    
    def denorm_coord(self, coord, min_val, max_val):
        return ((coord + 1) * (max_val - min_val) / 2) + min_val


    def forward(self, preds, target):

        input = self.denorm_coord(target, self.z_constraints_min, self.z_constraints_max)
        output = self.denorm_coord(preds, self.z_constraints_min, self.z_constraints_max)
        loss = F.mse_loss(input, output)
        return torch.mean(loss).float()
