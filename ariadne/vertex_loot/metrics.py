import torch 
import torch.nn as nn
import torch.nn.functional as F
import gin
import torchmetrics as tm

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


def _prepare_inputs_for_metrics(preds, target, station_constraints):
    orig_preds = _denorm_and_to_cartesian(preds.detach().cpu(), station_constraints)
    orig_target = _denorm_and_to_cartesian(target.detach().cpu(), station_constraints)
    return orig_preds, orig_target


@gin.configurable
class VertexMeanAbsoluteError(tm.Metric):
    def __init__(self, station_constraints, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.station_constraints = station_constraints
        self.__name__ = "mae"

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds, target = _prepare_inputs_for_metrics(preds, target, self.station_constraints)
        self.mae += torch.sum(torch.abs(preds - target))
        self.total += target.numel()

    def compute(self):
        return self.mae.float() / self.total


@gin.configurable
class VertexMeanSquaredError(tm.Metric):
    def __init__(self, station_constraints, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.station_constraints = station_constraints
        self.__name__ = "mse"

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds, target = _prepare_inputs_for_metrics(preds, target, self.station_constraints)
        self.mse += torch.sum(torch.square(preds - target))
        self.total += target.numel()

    def compute(self):
        return self.mse.float() / self.total