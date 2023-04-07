import torch
import gin
#import adamp

# optimizers
gin.external_configurable(torch.optim.Adam, denylist=['params'], name='Adam')
gin.external_configurable(torch.optim.AdamW, denylist=['params'], name='AdamW')
#gin.external_configurable(adamp.AdamP, denylist=['params'], name='AdamP')
gin.external_configurable(torch.optim.SGD, denylist=['params'], name='SGD')
gin.external_configurable(torch.optim.RMSprop, denylist=['params'], name='RMSprop')
gin.external_configurable(torch.optim.Adagrad, denylist=['params'], name='Adagrad')
gin.external_configurable(torch.optim.Adadelta, denylist=['params'], name='Adadelta')
gin.external_configurable(torch.optim.Adamax, denylist=['params'], name='Adamax')

gin.external_configurable(torch.nn.MSELoss, name='Mse_loss')
gin.external_configurable(torch.nn.L1Loss, name='Mae_loss')
gin.external_configurable(torch.nn.SmoothL1Loss, name='Mae_smooth_loss')

gin.external_configurable(torch.nn.BCELoss, name='BCE_loss')
gin.external_configurable(torch.nn.BCEWithLogitsLoss, name='BCEWithLogits_loss')

gin.external_configurable(torch.Tensor, name='tensor')
