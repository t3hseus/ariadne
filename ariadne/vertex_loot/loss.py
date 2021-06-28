import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


def denorm_coord(coord, min_val, max_val):
   return ((coord + 1) * (max_val - min_val) / 2) + min_val


def denorm(norm_vertex, x_constraints, y_constraints, z_constraints):
  x = denorm_coord(norm_vertex[:, 0], x_constraints[0], x_constraints[1])
  y = denorm_coord(norm_vertex[:, 1], y_constraints[0], y_constraints[1])
  z = denorm_coord(norm_vertex[:, 2], z_constraints[0], z_constraints[1])
  return torch.stack([x, y, z], dim=1)  



def to_cartesian(vertex):
  y_new = vertex[:, 0] * torch.cos(vertex[:, 1])
  x_new = vertex[:, 0] * torch.sin(vertex[:, 1])
  return torch.stack([x_new, y_new, vertex[:, 2]], dim=1) 


def _denorm_and_to_cartesian(coords, station_constraints):
    orig_coords = to_cartesian(
        denorm(coords, 
               x_constraints=station_constraints['vx'],
               y_constraints=station_constraints['vy'],
               z_constraints=station_constraints['vz'])
    )
    return orig_coords


@gin.configurable
class LossL2(nn.Module):

    def __init__(self, station_constraints):
        super(LossL2, self).__init__()
        self.station_constraints = station_constraints
        

    def forward(self, preds, target):
        input = _denorm_and_to_cartesian(preds.detach().cpu(), self.station_constraints)
        output = _denorm_and_to_cartesian(target.detach().cpu(), self.station_constraints)
        loss = F.mse_loss(input, output)
        return torch.mean(loss).float()
