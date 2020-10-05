import torch
import gin

# optimizers
gin.external_configurable(torch.optim.Adam, blacklist=['params'], name='Adam')
gin.external_configurable(torch.optim.SGD, blacklist=['params'], name='SGD')
gin.external_configurable(torch.optim.RMSprop, blacklist=['params'], name='RMSprop')
gin.external_configurable(torch.optim.Adagrad, blacklist=['params'], name='Adagrad')
gin.external_configurable(torch.optim.Adadelta, blacklist=['params'], name='Adadelta')
gin.external_configurable(torch.optim.Adamax, blacklist=['params'], name='Adamax')