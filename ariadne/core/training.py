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

    def _calc_loss_on_batch(self, batch):
        x, y = batch
        y_pred = self.model(x)
        return self.criterion(y_pred, y)

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss_on_batch(batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss_on_batch(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def configure_optimizers(self):
        return self.optimizer(self.parameters())