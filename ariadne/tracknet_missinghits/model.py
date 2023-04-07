import gin
import torch
import torch.nn as nn
import math
from ariadne.layers import CausalConv1d


ALLOWED_RNN_TYPES = ['GRU', 'LSTM']


@gin.configurable
class TrackNETMissingHits(nn.Module):
    """Builds TrackNETv2 model

    # Arguments
        input_features: number of input features (channels)
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
    """
    def __init__(self,
                 input_features=4,
                 hidden_features=32,
                 output_features=2,
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
            num_layers=2,
            batch_first=batch_first
        )
        
        self.coords_1 = nn.Sequential(
            nn.Linear(hidden_features, output_features)
        )
        self.radius_1 = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.Softplus()
        )
        self.coords_2 = nn.Sequential(
            nn.Linear(hidden_features, output_features)
        )
        self.radius_2 = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.Softplus()
        )
        

    def forward(self, inputs, input_lengths=None):
        input_lengths = input_lengths.int().cpu()
        x = inputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        coords_1 = self.coords_1(x)
        radius_1 = self.radius_1(x)
        coords_2 = self.coords_2(x)
        radius_2 = self.radius_2(x)
        outputs = torch.cat([coords_1, radius_1, coords_2, radius_2], dim=-1)
        return outputs
        """
        x = inputs
        x, _ = self.rnn(x)
        coords_1 = self.coords_1(x)
        radius_1 = self.radius_1(x)
        coords_2 = self.coords_2(x)
        radius_2 = self.radius_2(x)
        outputs = torch.cat([coords_1, radius_1, coords_2, radius_2], dim=-1)
        return outputs
        """