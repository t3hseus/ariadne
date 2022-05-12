import logging

import os
import traceback

#import _gin_bugfix
import gin
import numpy as np
import torch
import torch.nn as nn

from absl import flags
from absl import app

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ariadne.lightning import TrainModel
from ariadne.utils.model import get_checkpoint_path, weights_update

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
    fh.setFormatter(formatter)

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
               accumulate_grad_batches=1,
               resume_from_checkpoint=None,  # path to checkpoint to resume
               num_gpus=None,
               clip_grads=False
               ):

    os.makedirs(log_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(log_dir, name=model.__name__)
    setup_logger(tb_logger.log_dir)

    with open(os.path.join(tb_logger.log_dir, "train_config.cfg"), "w") as f:
        f.write(gin.config_str())

    LOGGER.info(f"Log directory {tb_logger.log_dir}")
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
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = get_checkpoint_path(model_dir=resume_from_checkpoint,
                                                 version=None,
                                                 checkpoint='latest')
        LOGGER.info(f"Resuming from checkpoint {resume_from_checkpoint}")

    LOGGER.info(model)
    if hasattr(model.data_loader, "get_num_train_events") and hasattr(model.data_loader, "get_num_val_events"):
        LOGGER.info(f"Number events for training {model.data_loader.get_num_train_events()}")
        LOGGER.info(f"Number events for validation {model.data_loader.get_num_val_events()}")
    # configure trainer
    trainer_kwargs = {
        'max_epochs': epochs,
        'auto_select_gpus': True,
        'deterministic': True,
        'terminate_on_nan': True,
        'accumulate_grad_batches': accumulate_grad_batches,
        'logger': tb_logger,
        'gpus': num_gpus
        #'progress_bar_refresh_rate': 100,
        #'log_every_n_steps': 50,
    }
    
    if clip_grads:
        trainer_kwargs['gradient_clip_val'] = 0.5

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{trainer_kwargs['logger'].log_dir}",
        filename=f'{{epoch}}-{{step}}'
    )
    trainer_kwargs['callbacks'] = [checkpoint_callback]
    if torch.cuda.is_available():
        trainer_kwargs['gpus'] = 1 if trainer_kwargs['gpus'] is None else trainer_kwargs['gpus']
        if trainer_kwargs['gpus'] > 1:
            # TODO: fix multi-GPU support
            # TODO: fix multi-GPU support
            trainer_kwargs['distributed_backend'] = 'ddp'
            trainer_kwargs['accelerator'] = 'ddp'
    else:
        trainer_kwargs['gpus'] = None
        if num_gpus is not None:
            LOGGER.warning(f"Warning! You set num_gpus={num_gpus} but torch.cuda.is_available() is '{torch.cuda.is_available()}'."
                           f"Training will be on the CPU!")

    if fp16_training:
        # TODO: use torch.nn.functional.binary_cross_entropy_with_logits which is safe to autocast
        trainer_kwargs['precision'] = 16
        trainer_kwargs['amp_level'] = '02'
    try:
        trainer = Trainer(resume_from_checkpoint=resume_from_checkpoint, **trainer_kwargs)
    except Exception as exc:
        #if one of keys is not in checkpoint etc
        LOGGER.error(f"Got exception! exc: {exc}. \n{traceback.format_exc()}")
        model.model = weights_update(model=model.model,
                           checkpoint=torch.load(resume_from_checkpoint))
        trainer = Trainer(resume_from_checkpoint=None, **trainer_kwargs)

    trainer.fit(model=model)

import  re
def main(argv):
    del argv
    FLAGS.config = './train_rd.cfg'
    gin.parse_config(open(FLAGS.config))

    LOGGER.setLevel(FLAGS.log)
    experiment()
    LOGGER.info("end training")
    return 0

    EPOCH_NUM = 2

    input_dir = './output/spd_sim_prepared/rd_fake_restriction'
    parts= [ input_dir + '/' + file_name for file_name in os.listdir(input_dir)  if  re.match(r"part[0-9]+", file_name) ]

    current_epoch = 1
    resume = False
    for i in range(EPOCH_NUM):

        for p in parts:

            gin.bind_parameter('GraphsDatasetMemory.input_dir', p)
            gin.bind_parameter('experiment.epochs',current_epoch)

            gin.bind_parameter('GraphDataLoader.n_train', len( os.listdir(p) ) - 2 )

            current_epoch+=1
            LOGGER.setLevel(FLAGS.log)
            if not resume:
                resume = True
                experiment()
            else:

                dirs = [d for d in os.listdir('./lightning_logs/GraphNet_v1') ]
                latest_dir = sorted(dirs, key=lambda x: os.path.getctime( './lightning_logs/GraphNet_v1/'+x), reverse=True)[:1]
                experiment( resume_from_checkpoint='./lightning_logs/GraphNet_v1/'+latest_dir[0] )
            LOGGER.info("end training")

if __name__ == '__main__':
    app.run(main)
