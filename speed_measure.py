import torch
import gin
import logging
#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
from torch.autograd.profiler import profile
from tqdm import tqdm
from absl import flags
from absl import app

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
def timeit(batch_size, n_epochs, model, device_name):
    LOGGER.info("Strting measurement")
    device = torch.device(device_name)
    use_cuda = True if device_name=='cuda' else False
    model = TrackNETv2().to(device)
    with profile(use_cuda=use_cuda) as prof:
        for _ in tqdm(range(n_epochs)):
            temp = torch.rand(batch_size, 2, model.input_features).to(device)
            temp_lengths = (torch.ones(batch_size) * 2).to(device)
            #print(temp_lengths)
            preds = model(inputs=temp, input_lengths=temp_lengths)

    table = prof.key_averages().table()
    print(table)
    result = 'Speed:', round((batch_size * n_epochs) / float(str(table).split('\n')[-2].split(' ')[-1].strip('s')), 3), 'elements/s'
    LOGGER.info(table)
    LOGGER.info(result)
    print(result)

def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    timeit()

if __name__ == '__main__':
    app.run(main)