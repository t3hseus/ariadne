import logging

import os

import gin
import numpy as np
import torch
import torch.nn as nn

from absl import flags
from absl import app

from pytorch_lightning import Trainer, seed_everything
from ariadne.lightning import TrainModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

def setup_logger(logger_dir):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug

    os.makedirs(logger_dir , exist_ok=True)
    fh = logging.FileHandler('%s/train.log' % logger_dir)
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    #ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(fh)

LOGGER = logging.getLogger('ariadne.train')

@gin.configurable
def experiment(model,
               criterion,
               metrics,
               optimizer,
               data_loader,
               epochs,
               log_dir='lightning_logs',
               fp16_training=False,
               random_seed=None,
               accumulate_grad_batches=1):

    os.makedirs(log_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(log_dir, name=model.__name__)
    setup_logger(tb_logger.log_dir)


    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())

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
        'terminate_on_nan': True,
        'accumulate_grad_batches': accumulate_grad_batches,
        'logger':tb_logger
        #'accumulate_grad_batches' : 5
    }
    trainer_kwargs['checkpoint_callback'] = ModelCheckpoint(
        # TODO: check epoch and step
        filepath=f"{trainer_kwargs['logger'].log_dir}/_{{epoch}}-{{step}}")
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
    experiment()
    LOGGER.info("end training")

if __name__ == '__main__':
    app.run(main)
