import torch 
import torch.nn as nn
import torch.nn.functional as F
import gin
import torchmetrics as tm

def denorm_coord(coord, min_val, max_val):
   return ((coord + 1) * (max_val - min_val) / 2) + min_val


def _Zprepare_inputs_for_metrics(preds, target, station_constraints):
    orig_preds = denorm_coord(preds.detach().cpu(),
        min_val=station_constraints['vz'][0],
        max_val=station_constraints['vz'][1]
    )
    orig_target = denorm_coord(
        target.detach().cpu(),
        min_val=station_constraints['vz'][0],
        max_val=station_constraints['vz'][1]
    )
    return orig_preds, orig_target


@gin.configurable
class ZVertexMeanAbsoluteError(tm.Metric):
    def __init__(self, station_constraints, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.station_constraints = station_constraints
        self.__name__ = "mae"

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        f = open('./text.txt', 'w')
        f.write(str(preds.shape))
        f.write("_____")
        f.write(str(target.shape))
        f.close()
        assert preds.shape == target.shape
        preds, target = _Zprepare_inputs_for_metrics(preds, target, self.station_constraints)
        self.mae += torch.sum(torch.abs(preds - target))
        self.total += target.numel()

    def compute(self):
        return self.mae.float() / self.total


@gin.configurable
class ZVertexMeanSquaredError(tm.Metric):
    def __init__(self, station_constraints, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.station_constraints = station_constraints
        self.__name__ = "mse"

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        f = open('./text.txt', 'w')
        f.write(str(preds.shape))
        f.write("_____")
        f.write(str(target.shape))
        f.close()
        assert preds.shape == target.shape
        preds, target = _Zprepare_inputs_for_metrics(preds, target, self.station_constraints)
        self.mse += torch.sum(torch.square(preds - target))
        self.total += target.numel()

    def compute(self):
        return self.mse.float() / self.total