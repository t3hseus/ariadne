import logging
import gin
import torch
import torch.nn as nn

from absl import flags
from absl import app
from ariadne.core.training import TrainModel

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

LOGGER = logging.getLogger('ariadne.train')

@gin.configurable
def experiment(model,
               criterion,
               metrics,
               optimizer):
    LOGGER.info('Create model for training')
    # create model for trainingimport logging
    model = TrainModel(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer
    )
    LOGGER.info(model)


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    experiment()


if __name__ == '__main__':
    app.run(main)