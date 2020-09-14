import gin
import torch
import torch.nn as nn
import pytorch_lightning as pl

# external gin configurable
gin.external_configurable(torch.optim.Adam, blacklist=['params'], name='Adam')
gin.external_configurable(torch.optim.SGD, blacklist=['params'], name='SGD')
gin.external_configurable(torch.optim.RMSprop, blacklist=['params'], name='RMSprop')
gin.external_configurable(torch.optim.Adagrad, blacklist=['params'], name='Adagrad')
gin.external_configurable(torch.optim.Adadelta, blacklist=['params'], name='Adadelta')
gin.external_configurable(torch.optim.Adamax, blacklist=['params'], name='Adamax')


class TrainModel(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 metrics,
                 optimizer):
        super().__init__()
        # configure model and criterion
        self.model = model()
        self.criterion = criterion()
        self.optimizer = optimizer
        self.metrics = metrics

    def forward(self, inputs):
        """
        # Arguments
            inputs (dict): kwargs dict with model inputs
        """
        return self.model(**inputs)

    def _calc_metrics(self, preds, target):
        metric_vals = {}
        for metric in self.metrics:
            metric_vals[metric.__name__] = metric(preds, target)
        return metric_vals

    def _forward_batch(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        metric_vals = self._calc_metrics(y_pred, y)
        return {'loss': loss, **metric_vals}

    def training_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'train_{k}': v for k, v in result_dict}
        result = pl.TrainResult(result_dict['loss'])
        result.log_dict(tqdm_dict, on_step=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f'val_{k}': v for k, v in result_dict}
        result = pl.EvalResult(checkpoint_on=result_dict['loss'])
        result.log_dict(tqdm_dict)
        return result

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
