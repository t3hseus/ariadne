import math
import os
from dataclasses import dataclass
from typing import Collection

import gin
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ariadne.point_net.point.points import Points, load_points


class MemoryDataset(Dataset):
    def __init__(self, input_dir):
        super(MemoryDataset, self).__init__()
        self.input_dir = input_dir

    def load_data(self):
        raise NotImplementedError


class ItemLengthGetter(object):
    def __init__(self):
        pass

    def get_item_length(self, item_index):
        raise NotImplementedError


@gin.configurable
class CloudDataset(MemoryDataset, ItemLengthGetter):
    def __init__(self, input_dir, n_samples=None, pin_mem=False, sample_len=2048):
        super().__init__(os.path.expandvars(input_dir))
        self.n_samples = n_samples
        self.filenames = []
        self.points = {}
        self.pin_mem = pin_mem
        self.sample_len = sample_len

        filenames = [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.endswith(".npz")
        ]
        self.filenames = (
            filenames[: self.n_samples] if self.n_samples is not None else filenames
        )

    def __getitem__(self, index):
        if self.pin_mem:
            if index in self.points:
                item = self.points[index]
            else:
                item = self.points[index] = load_points(self.filenames[index])

            return item
        return load_points(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


class SubsetWithItemLen(Subset, ItemLengthGetter):
    def __init__(self, dataset, indices):
        super(SubsetWithItemLen, self).__init__(dataset, indices)

    def get_item_length(self, item_index):
        return len(self.dataset[self.indices[item_index]].track)


@dataclass
class GridConfigNew:
    # points cloud range
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    zrange: tuple[float, float]
    vx: float = 0.4
    vy: float = 0.2
    vz: float = 0.2

    def __post_init__(self):
        # voxel grid
        self.W = math.ceil((self.xrange[1] - self.xrange[0]) / self.vx)
        self.H = math.ceil((self.yrange[1] - self.yrange[0]) / self.vy)
        self.D = math.ceil((self.zrange[1] - self.zrange[0]) / self.vz)


@gin.configurable("cloud_collate_fn")
def collate_fn(points: Collection[Points], sample_len=512, shuffle=True):
    batch_size = len(points)
    n_feat = 3
    n_dim = np.array([(min(p.X.shape[1], sample_len)) for p in points])
    max_dim = sample_len  # n_dim.max()
    batch_inputs = np.zeros((batch_size, n_feat, max_dim), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_dim), dtype=np.float32)
    batch_targets = np.zeros((batch_size, max_dim), dtype=np.float32)
    for i, p in enumerate(points):
        batch_mask[i, :n_dim[i]] = 1.0
        perm_ids = np.arange(0, n_dim[i])
        assert perm_ids[-1] == n_dim[i] - 1, "You need to specify upper bound!"
        if shuffle:
            perm_ids = np.random.permutation(n_dim[i])

        batch_inputs[i, :, : n_dim[i]] = p.X[:, perm_ids]
        batch_targets[i, : n_dim[i]] = np.float32(p.track[perm_ids] != -1)
    # max_dim = 512
    # batch_inputs = batch_inputs[:, :, :max_dim]
    # batch_targets = batch_targets[:, :max_dim]
    # batch_mask = batch_mask[:, :max_dim]
    batch_inputs = np.swapaxes(batch_inputs, -1, -2)
    batch_targets = np.expand_dims(batch_targets, -1)
    x = torch.from_numpy(batch_inputs)
    return (
        {"x": x, "mask": torch.from_numpy(batch_mask)},
        {"y": torch.from_numpy(batch_targets), "mask": torch.from_numpy(batch_mask)},
    )


@gin.configurable("voxel_collate_fn")
def voxel_collate_fn(
    points: Collection[Points],
    sample_len=512,
    shuffle=True,
    xrange=(-1.0, 1.0),
    yrange=(-1.0, 1.0),
    zrange=(-1.0, 1.0),
    vx=0.5,
    vy=0.5,
    vz=0.5,
):
    cfg = GridConfigNew(
        xrange=xrange,
        yrange=yrange,
        zrange=zrange,
        vx=vx,
        vy=vy,
        vz=vz,
    )
    batch_inputs = []
    batch_targets = []
    batch_masks = []

    for i, p in enumerate(points):
        perm_ids = np.arange(0, len(p.track))
        assert perm_ids[-1] == len(p.track) - 1, "You need to specify upper bound!"
        if shuffle:
            perm_ids = np.random.permutation(len(p.track))
        batch = p.X[:, perm_ids].T

        targets = (p.track[perm_ids] != -1).astype(np.float32)
        batch_dict = process_pointcloud(
            point_cloud=batch, targets=targets, cfg=cfg, max_points_per_voxel=sample_len
        )
        batch_inputs.extend(batch_dict["x"])
        batch_targets.extend(batch_dict["labels"])
        batch_masks.extend(batch_dict["mask"])
    batch_inputs = np.array(batch_inputs)
    batch_masks = np.array(batch_masks)
    batch_targets = np.array(batch_targets).astype(np.float32)
    # batch_inputs = np.swapaxes(batch_inputs, -1, -2)
    batch_targets = np.expand_dims(batch_targets, -1)
    # max_dim = 512
    # batch_inputs = batch_inputs[:, :, :max_dim]
    # batch_targets = batch_targets[:, :max_dim]
    # batch_mask = batch_mask[:, :max_dim]

    x = torch.from_numpy(batch_inputs)

    return (
        {"x": x, "mask": torch.from_numpy(batch_masks)},
        {"y": torch.from_numpy(batch_targets), "mask": torch.from_numpy(batch_masks)},
    )


def process_pointcloud(
    point_cloud,
    targets,
    cfg: GridConfigNew,
    max_points_per_voxel: int = 512,
) -> dict[str, list[np.array]]:
    voxel_features = []
    voxel_targets = []
    voxel_masks = []
    grid_coords = (
        (point_cloud - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]]))
        / (cfg.vx, cfg.vy, cfg.vz)
    ).astype(np.int32)
    voxel_coords, inv_ind, voxel_counts = np.unique(
        grid_coords, axis=0, return_inverse=True, return_counts=True
    )

    for i in range(len(voxel_coords)):
        voxel = np.zeros((max_points_per_voxel, 6), dtype=np.float32)
        padded_voxel_targets = np.zeros(max_points_per_voxel, dtype=np.int)
        mask = np.zeros(max_points_per_voxel, dtype=np.int)

        pts = point_cloud[inv_ind == i]
        this_voxel_targets = targets[inv_ind == i]

        if voxel_counts[i] > max_points_per_voxel:
            pts = pts[:max_points_per_voxel, :]
            targets = targets[:max_points_per_voxel]
            mask[:max_points_per_voxel] = 1
            voxel_counts[i] = max_points_per_voxel
        # augment the points
        voxel[: pts.shape[0], :] = np.concatenate(
            (pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1
        )
        mask[: pts.shape[0]] = 1.0
        padded_voxel_targets[: pts.shape[0]] = this_voxel_targets
        voxel_features.append(voxel)
        voxel_targets.append(padded_voxel_targets)
        voxel_masks.append(mask)
    return {
        "x": voxel_features,
        "labels": voxel_targets,
        "mask": voxel_masks,
        "voxel_coords": voxel_coords,
    }


if __name__ == "__main__":
    point_cloud = np.random.normal(1, 1, (100, 3)).clip(0, 1.99)
    labels = np.random.binomial(100, 0.3, 100)
    mask = np.ones(100)
    # config = GridConfig(
    #     scene_size=[2., 2., 2.],
    #     voxel_size=[0.5, 0.5, 0.5],
    #     max_point_number=50
    # )
    config = GridConfigNew(
        xrange=(0.0, 2.0), yrange=(0.0, 2.0), zrange=(0.0, 2.0), vd=0.5, vh=0.5, vw=0.5
    )
    # print(point_cloud)

    voxel_dict = process_pointcloud(
        point_cloud=point_cloud, cfg=config, targets=labels, max_points_per_voxel=50
    )
    # print(voxel_dict)
    for k, i in voxel_dict.items():
        print(k, [item.shape for item in i])
    print("uniform")
    point_cloud = np.random.uniform(0, 1.99, (2, 100, 3))
    labels = np.random.binomial(200, 0.3, (2, 100))
    mask = np.ones((2, 100))
    config = GridConfig(
        scene_size=[2.0, 2.0, 2.0], voxel_size=[0.5, 0.5, 0.5], max_point_number=50
    )

    voxel_dict = process_pointcloud(point_cloud=point_cloud, cfg=config, labels=labels)
    for k, i in voxel_dict.items():
        print(k, i.shape)
