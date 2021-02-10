import logging
import sys
import os
sys.path.append(os.path.abspath('./'))

import pandas as pd
import numpy as np
import torch
import gin

from tqdm import tqdm
from absl import flags
from absl import app

from torch.utils.data import DataLoader
from torch.autograd.profiler import profile

from ariadne.graph_net.graph_utils.graph import Graph
from ariadne.graph_net.model import GraphNet_v1
from ariadne.tracknet_v2.model import TrackNETv2
from ariadne.tracknet_v2.dataset import TrackNetV2Dataset, TrackNetV2ExplicitDataset
from ariadne.tracknet_v2.metrics import point_in_ellipse

LOGGER = logging.getLogger('ariadne.moments_histogram')
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)

def find_nearest_hit(hits, ellipse):
    center_x = ellipse[0]
    center_y = ellipse[1]
    dists = (hits['x'] - center_x) **2 + (hits['y'] - center_y) **2
    minimal = hits.reset_index().iloc[int(np.argmin(dists)),-1]
    return minimal

@gin.configurable
def create_hist(batch_size, device_name='cpu', model=None, n_epochs=1000, n_stations=2, half_precision=False, path_to_data=None):
    LOGGER.info("Starting measurement")
    LOGGER.info(f"No. CUDA devices: {torch.cuda.device_count()}")
    print(device_name)
    device = torch.device(device_name)
    use_cuda = True if device_name=='cuda' else False
    #if model is None:
    model = TrackNETv2(input_features=3).to(device).double()
    if half_precision:
        model = model.half()
    model.eval()
    #data = pd.read_table('resources/test_data/data_with_moment.txt', sep='\t', names=['event','x','y','z','station','track','px','py','pz','vx','vy','vz'])
    data = TrackNetV2ExplicitDataset(data_file='output/cgem_t_plain_explicit/tracknet_all.npz')
    last_station_hits = torch.tensor(np.load('output/cgem_t_plain_explicit/tracknet_all_last_station.npz')['hits']).to(device)
    print(last_station_hits.size())
    test_loader = TrackNetV2ExplicitDataLoader(dataset=data, batch_size=batch_size)
    with profile(use_cuda=use_cuda) as prof:
        for _ in tqdm(range(n_epochs)):
            for i, data in enumerate(test_loader, 0):
                print(data)
                lengths = data['x']['input_lengths'].to(torch.double).to(device)
                inputs = data['x']['inputs'].to(torch.double).to(device)
                reconstruction = model(inputs, lengths)
                #print(reconstruction)
                filter = data['is_real_track']
                print(reconstruction.size())
                real_reconstruction = torch.squeeze(reconstruction[torch.nonzero(filter)], dim=1)
                print(real_reconstruction.size())

    table = prof.key_averages().table()
    print(table)
    result = 0
    LOGGER.info(table)
    LOGGER.info(result)
    print(result)

def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("CONFIG: %s" % gin.config_str())
    create_hist()

if __name__ == '__main__':
    app.run(main)