import gin
import torch
import torch.nn as nn
import math
import numpy as np

ALLOWED_RNN_TYPES = ['GRU', 'LSTM']

@gin.configurable
class Selector(nn.Module):
    """Builds Selector model

    # Arguments
        input_features: number of input features (channels)
        hidden_features: number of hidden features
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
    """
    def __init__(self,
                 input_features=14,
                 hidden_features=128,
                 rnn_type='gru',
                 batch_first=True):
        super().__init__()
        self.input_features = input_features
        rnn_type = rnn_type.upper()
        if rnn_type not in ALLOWED_RNN_TYPES:
            raise ValueError(f'RNN type {rnn_type} is not supported. '
                             f'Choose one of {ALLOWED_RNN_TYPES}')
        _rnn_layer = getattr(nn, rnn_type)
        self.rnn = _rnn_layer(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=4,
            batch_first=batch_first
        )
        # outputs
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features, 2),
        )

    def forward(self, inputs, input_lengths, return_gru_states=False):
        input_lengths = input_lengths.int().cpu()
        x = inputs
        
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, enforce_sorted=False, batch_first=True)
        # forward pass trough rnn
        x, _ = self.rnn(packed)
        # unpack padding
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #print(x.shape)
        # get result using only the output on the last timestep
        output = self.classifier(x)

        return output[:, -1]
