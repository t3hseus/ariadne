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
               fp16_training=False,
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

    # configure trainer
    trainer_kwargs = {
        'max_epochs': epochs,
        'auto_select_gpus': True,
        'deterministic': True,
        'terminate_on_nan': True
    }
    if torch.cuda.is_available():
        trainer_kwargs['gpus'] = torch.cuda.device_count()
        if trainer_kwargs['gpus'] > 1:
            # TODO: fix multi-GPU support
            trainer_kwargs['distributed_backend'] = 'ddp'

    if fp16_training:
        # TODO: use torch.nn.functional.binary_cross_entropy_with_logits which is safe to autocast
        trainer_kwargs['precision'] = 16
        trainer_kwargs['amp_level'] = '02'

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model=model)


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())
    experiment()


if __name__ == '__main__':
    app.run(main)
