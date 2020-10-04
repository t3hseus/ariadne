import logging
import gin
import numpy as np
import torch
import torch.nn as nn

from absl import flags
from absl import app
from pytorch_lightning import Trainer, seed_everything

from ariadne.lightning import TrainModel

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
               optimizer,
               data_loader,
               epochs,
               random_seed=None):
    # for reproducibility
    if random_seed is not None:
        LOGGER.info('Setting random seed to %d', random_seed)
        seed_everything(random_seed)

    LOGGER.info('Create model for training')
    # create model for trainingimport logging
    model = TrainModel(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        data_loader=data_loader
    )

    LOGGER.info(model)

    trainer = Trainer(
        max_epochs=epochs, 
        deterministic=True
    )
    trainer.fit(model=model)


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    experiment()


if __name__ == '__main__':
    app.run(main)
