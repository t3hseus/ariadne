from typing import Union, List, Any

import gin
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ariadne.data_loader import BaseDataLoader


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
        print(torch.cuda.current_device())

    def forward(self, inputs):
        """
        # Arguments
            inputs (dict): kwargs dict with model inputs
        """
        print(torch.cuda.current_device())
        return self.model(**inputs)

    def _calc_metrics(self, batch_output, batch_target):
        metric_vals = {}
        for metric in self.metrics:
            metric_vals[metric.__name__] = metric(batch_output, batch_target)
        return metric_vals

    def _forward_batch(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        metric_vals = self._calc_metrics(y_pred, y)
        return {'loss': loss, **metric_vals}

    def training_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'train_{k}': v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, on_step=False, prog_bar=True)
        return tqdm_dict['train_loss']

#    @property
#    def example_input_array(self) -> Any:
#        return [self.data_loader.get_one_sample()]

    def validation_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'val_{k}': v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True)
        return tqdm_dict['val_loss']

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def train_dataloader(self):
        return self.data_loader.get_train_dataloader()

    def val_dataloader(self):
        return self.data_loader.get_val_dataloader()
