import logging
import gin
import torch
import torch.nn as nn

from ariadne.core.training import TrainModel

logging.basicConfig()
LOGGER = logging.getLogger('ariadne')


@gin.configurable
def experiment(model,
               criterion,
               metrics,
               optimizer):
    LOGGER.info('Create model for training')
    # create model for training
    model = TrainModel(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer
    )
    LOGGER.info(model)


def main():
    gin.parse_config(open('resources/gin/tracknet_v2_train.cfg'))
    LOGGER.setLevel('INFO')
    experiment()


if __name__ == '__main__':
    main()