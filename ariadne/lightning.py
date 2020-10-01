import gin
import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning.metrics.functional

import pytorch_lightning as pl

# external gin configurable
from ariadne.data_loader import BaseDataLoader

gin.external_configurable(torch.optim.Adam, blacklist=['params'], name='Adam')
gin.external_configurable(torch.optim.SGD, blacklist=['params'], name='SGD')
gin.external_configurable(torch.optim.RMSprop, blacklist=['params'], name='RMSprop')
gin.external_configurable(torch.optim.Adagrad, blacklist=['params'], name='Adagrad')
gin.external_configurable(torch.optim.Adadelta, blacklist=['params'], name='Adadelta')
gin.external_configurable(torch.optim.Adamax, blacklist=['params'], name='Adamax')

gin.external_configurable(torch.nn.BCELoss, whitelist=[], name='BCE_LOSS')


gin.external_configurable(pytorch_lightning.metrics.functional.accuracy, whitelist=['num_classes'], name='accuracy_score')
gin.external_configurable(pytorch_lightning.metrics.functional.recall, whitelist=[], name='recall_score')
gin.external_configurable(pytorch_lightning.metrics.functional.precision, whitelist=[], name='precision_score')
gin.external_configurable(pytorch_lightning.metrics.functional.f1_score, whitelist=[], name='f1_score')


class TrainModel(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 metrics,
                 optimizer,
                 data_loader: BaseDataLoader.__class__):
        super().__init__()
        # configure model and criterion
        self.model = model()
        self.criterion = criterion()
        self.optimizer = optimizer
        self.metrics = metrics
        self.data_loader = data_loader()

    def forward(self, inputs):
        """
        # Arguments
            inputs (dict): kwargs dict with model inputs
        """
        return self.model(**inputs)

    def _calc_metrics(self, batch_output, batch_target):
        metric_vals = {}
        # TODO: PLEASE refactor this!!
        #arr1 = ((batch_target > 0.5).cpu().numpy() != 0)
        #arr2 = ((batch_output > 0.5).cpu().numpy() != 0)
        for metric in self.metrics:
            metric_vals[metric.__name__] = metric(batch_output, batch_target)
        return metric_vals

    def _forward_batch(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        metric_vals = self._calc_metrics(y_pred, y)
        return {'loss': loss, **metric_vals}

    def training_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'train_{k}': v for k, v in result_dict.items()}
        result = pl.TrainResult(result_dict['loss'])
        result.log_dict(tqdm_dict, on_step=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'val_{k}': v for k, v in result_dict.items()}
        result = pl.EvalResult(checkpoint_on=result_dict['loss'])
        result.log_dict(tqdm_dict)
        return result

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def train_dataloader(self):
        return self.data_loader.get_train_dataloader()

    def val_dataloader(self):
        return self.data_loader.get_val_dataloader()
