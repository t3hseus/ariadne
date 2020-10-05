import logging
import sys
import os
sys.path.append(os.path.abspath('./'))

import torch
import gin

from torch.autograd.profiler import profile
from tqdm import tqdm
from absl import flags
from absl import app

from ariadne.graph_net.graph_utils.graph import Graph
from ariadne.graph_net.model import GraphNet_v1
from ariadne.tracknet_v2.model import TrackNETv2

LOGGER = logging.getLogger('ariadne.speed_measure')
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

@gin.configurable
def timeit_tracknet(batch_size, model, device_name, n_epochs=1000, n_stations=2, half_precision=False):
    LOGGER.info("Starting measurement")
    LOGGER.info(f"No. CUDA devices: {torch.cuda.device_count()}")
    device = torch.device(device_name)
    use_cuda = True if device_name=='cuda' else False
    model = TrackNETv2().to(device)
    if half_precision:
        model = model.half()
    model.eval()
    with profile(use_cuda=use_cuda) as prof:
        for _ in tqdm(range(n_epochs)):
            temp = torch.rand(batch_size, n_stations, model.input_features).to(device)
            temp_lengths = (torch.ones(batch_size) * n_stations).to(device)
            if half_precision:
                temp = temp.half()
                temp_lengths = temp_lengths.half()
            #print(temp_lengths)
            preds = model(inputs=temp, input_lengths=temp_lengths)

    table = prof.key_averages().table()
    print(table)
    result = 'Speed:', round((batch_size * n_epochs) / float(str(table).split('\n')[-2].split(' ')[-1].strip('s')), 3), 'elements/s'
    LOGGER.info(table)
    LOGGER.info(result)
    print(result)


@gin.configurable
def timeit_graph(batch_size, model, device_name, n_epochs=1000, n_stations=2, half_precision=False):
    LOGGER.info("Starting measurement")
    LOGGER.info(f"No. CUDA devices: {torch.cuda.device_count()}")
    device = torch.device(device_name)
    use_cuda = True if device_name=='cuda' else False
    model = GraphNet_v1().to(device)
    if half_precision:
        model = model.half()
    model.eval()
    N = 900
    E = 1300
    F = 5
    with profile(use_cuda=use_cuda) as prof:
        for _ in tqdm(range(n_epochs)):
            temp_X = torch.rand(batch_size, N, F)
            temp_Ri = torch.rand(batch_size, N, E)
            temp_Ro = temp_Ri

            if half_precision:
                temp_X = temp_X.half()
                temp_Ri = temp_X.half()
                temp_Ro = temp_X.half()
            #print(temp_lengths)
            graph = (temp_X, temp_Ri, temp_Ro)
            preds = model(graph)

    table = prof.key_averages().table()
    print(table)
    result = 'Speed:', round((batch_size * n_epochs) / float(str(table).split('\n')[-2].split(' ')[-1].strip('s')), 3), 'elements/s'
    LOGGER.info(table)
    LOGGER.info(result)
    print(result)


@gin.configurable
def experiment(timeit_func):
    timeit_func()


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("CONFIG: %s" % gin.config_str())
    experiment()

if __name__ == '__main__':
    app.run(main)