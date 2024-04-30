import numpy as np
import torch
from enum import IntEnum
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ariadne.transformer.spd_generator import SPDEventGenerator
from typing import Tuple, Any


class DatasetMode(IntEnum):
    train = 0
    val = 1
    test = 2


def time_slice_collator(x):
    return x[0]  # only one timeslice in a batch

class Normalizer:
    """ MinMax scaler in range from -1 to 1
    for each coordinate
    """
    def __init__(
        self,
        x_coord_range: Tuple[float, float] = (-851., 851.),
        y_coord_range: Tuple[float, float] = (-851., 851.),
        z_coord_range: Tuple[float, float] = (-2386., 2386.)
    ):
        self.x_min, self.x_max = x_coord_range
        self.y_min, self.y_max = y_coord_range
        self.z_min, self.z_max = z_coord_range

    def __call__(
        self,
        hits: np.ndarray[(Any, 3), np.float32]
    ) -> np.ndarray[(Any, 3), np.float32]:
        x_norm = 2 * (hits[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (hits[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
        z_norm = 2 * (hits[:, 2] - self.z_min) / (self.z_max - self.z_min) - 1
        normalized_hits = np.hstack([
            x_norm.reshape(-1, 1),
            y_norm.reshape(-1, 1),
            z_norm.reshape(-1, 1)
        ])
        return normalized_hits


class SPDTimesliceTracksDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        detector_eff: float = 0.98,
        n_events_timeslice: int = 40,
        hits_normalizer: Optional[Callable] = None,
        mode: DatasetMode = DatasetMode.train
    ):
        self.spd_gen = SPDEventGenerator(
            n_events_timeslice=n_events_timeslice,
            detector_eff=detector_eff,
            add_fakes=True
        )
        self._n_samples = n_samples
        self.hits_normalizer = hits_normalizer
        # get initial random seed for reproducibility
        # mode helps to ensure that datasets don't intersect
        self._initial_seed = np.random.get_state()[1][mode]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        
        # prevent dataset from generation of new samples each epoch
        np.random.seed(self._initial_seed + idx)
        # generate sample
        time_slice = self.spd_gen.generate_spd_event()

        hits = np.concatenate([time_slice['hits'], time_slice['fakes']]).T
        labels = np.concatenate([time_slice["track_ids"], np.full((time_slice['fakes'].shape[0],), fill_value = -1, dtype = int)])
        
        if self.hits_normalizer:
            hits = self.hits_normalizer(hits)
        
        norm = Normalizer()
        hits = norm(hits)
            
        n_tracks = np.unique(time_slice["track_ids"]).shape[0]

        return hits, labels, n_tracks
