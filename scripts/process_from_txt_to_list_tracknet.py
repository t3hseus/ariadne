import logging
from typing import List, Tuple, Optional, Iterable
from ariadne.preprocessing import DataProcessor, DataChunk, ProcessedDataChunk, ProcessedData
from ariadne.tracknet_v2.model import TrackNETv2
import gin
import pandas as pd
import numpy as np
import itertools
from ariadne.transformations import Compose, StandardScale, ToCylindrical, \
    ConstraintsNormalize, MinMaxScale, DropSpinningTracks, DropFakes, DropShort
from tqdm import tqdm
from copy import deepcopy
from typing import Dict
LOGGER = logging.getLogger('ariadne.prepare')
import torch
from ariadne.tracknet_v2.metrics import point_in_ellipse
from ariadne.parsing import parse_df
from prepare import parse
from ariadne.utils import get_fake_tracks_from_two_first_stations

from absl import flags
from absl import app
FLAGS = flags.FLAGS

@gin.configurable()
def process_input_data_to_lists(file_name,
                 csv_params: Dict[str, object],
                 events_quantity=10,
                 transformer=None,
                 radial_stations_constraints=None):
    data_df = parse_df(file_name, **csv_params)
    res = np.arange(int(events_quantity))
    data_df = data_df[data_df.event.isin(res)]
    inputs = []
    ys = []
    lengths = []
    reals = []
    momentums = []
    events = []
    last_stations = []
    last_station_events = []

    if transformer is None:
        transformer = Compose([
            DropSpinningTracks(),
            DropShort(),
            ToCylindrical(),
            ConstraintsNormalize(
                use_global_constraints=False,
                constraints=radial_stations_constraints,
                columns=('r', 'phi', 'z')
            ),
        ])
    processed = transformer(data_df)
    chunk_df = processed.groupby('event')
    first = 1
    for id, df in chunk_df:
        chunk_data_x = []
        chunk_data_y = []
        chunk_data_len = []
        chunk_data_real = []
        chunk_data_momentum = []
        grouped_df = df[df['track'] != -1].groupby('track')
        last_station = df[df['station'] > 1][['phi', 'z']].values
        for i, data in grouped_df:
            chunk_data_x.append(data[['r', 'phi', 'z']].values[:-1])
            chunk_data_y.append(data[['phi', 'z']].values[-1])
            chunk_data_len.append(2)
            chunk_data_momentum.append(data[['px', 'py', 'pz']].values[-1])
            chunk_data_real.append(1)
        first_station = df[df['station'] == 0][['r', 'phi', 'z', 'track']]
        first_station.columns = ['r_left', 'phi_left', 'z_left', 'track_left']

        second_station = df[df['station'] == 1][['r', 'phi', 'z', 'track']]
        second_station.columns = ['r_right', 'phi_right', 'z_right', 'track_right']

        fake_tracks = get_fake_tracks_from_two_first_stations(first_station, second_station)
        fake_tracks = fake_tracks.sample(n=int(fake_tracks.shape[0] / 10), random_state=1)
        for i, row in fake_tracks.iterrows():
            temp_data = np.zeros((2, 3))
            temp_data[0, :] = row[['r_left', 'phi_left', 'z_left']].values
            temp_data[1, :] = row[['r_right', 'phi_right', 'z_right']].values
            chunk_data_x.append(temp_data)
            chunk_data_y.append(chunk_data_y[0])
            chunk_data_momentum.append(chunk_data_momentum[0])
            chunk_data_real.append(0)
            chunk_data_len.append(2)

        chunk_data_x = np.stack(chunk_data_x, axis=0)
        chunk_data_y = np.stack(chunk_data_y, axis=0)
        chunk_data_momentum = np.stack(chunk_data_momentum, axis=0)
        chunk_data_real = np.stack(chunk_data_real, axis=0)
        chunk_data_len = np.stack(chunk_data_len, axis=0)
        chunk_data_event = id
        chunk_data_event_last_station = id

        inputs.append(chunk_data_x)
        ys.append(chunk_data_y)
        reals.append(chunk_data_real)
        lengths.append(chunk_data_len)
        momentums.append(chunk_data_momentum)
        events.append(chunk_data_event)
        last_stations.append(last_station)
        last_station_events.append(chunk_data_event_last_station)
    return inputs, ys, reals, lengths, momentums, events, last_stations, last_station_events

def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    process_input_data_to_lists()

if __name__ == '__main__':
    app.run(main)


