import torch
import gin

# optimizers
gin.external_configurable(torch.optim.Adam, denylist=['params'], name='Adam')
gin.external_configurable(torch.optim.SGD, denylist=['params'], name='SGD')
gin.external_configurable(torch.optim.RMSprop, denylist=['params'], name='RMSprop')
gin.external_configurable(torch.optim.Adagrad, denylist=['params'], name='Adagrad')
gin.external_configurable(torch.optim.Adadelta, denylist=['params'], name='Adadelta')
gin.external_configurable(torch.optim.Adamax, denylist=['params'], name='Adamax')