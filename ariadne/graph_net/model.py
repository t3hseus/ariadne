import gin
import torch
import torch.nn as nn


ALLOWED_RNN_TYPES = ['GRU', 'LSTM']

@gin.configurable
class RDGraphNet(nn.Module):
    """Builds Reversed Directed Graph Neural Network model

    # Arguments
        input_features: number of input features (channels)
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
    """
    def __init__(self, 
                 input_features=4, 
                 conv_features=32,
                 rnn_type='gru',
                 batch_first=True):
        super().__init__() 
        rnn_type = rnn_type.upper()
        if rnn_type not in ALLOWED_RNN_TYPES:
            raise ValueError(f'RNN type {rnn_type} is not supported. '
                             f'Choose one of {ALLOWED_RNN_TYPES}')
        _rnn_layer = getattr(nn, rnn_type)
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, conv_features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(conv_features)
        )
        self.rnn = _rnn_layer(
            input_size=conv_features,
            hidden_size=conv_features,
            num_layers=2,
            batch_first=batch_first
        )
        # outputs
        self.xy_coords = nn.Sequential(
            nn.Linear(conv_features, 2)
        )
        self.r1_r2 = nn.Sequential(
            nn.Linear(conv_features, 2),
            nn.Softplus()
        )

    def forward(self, inputs, input_lengths):
        # BxTxC -> BxCxT
        inputs = inputs.transpose(1, 2)
        x = self.conv(inputs)
        # BxCxT -> BxTxC
        x = x.transpose(1, 2)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=True, batch_first=True)
        # forward pass trough rnn
        x, _ = self.rnn(packed)
        # unpack padding
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # get result using only the output on the last timestep
        xy_coords = self.xy_coords(x[:, -1])
        r1_r2 = self.r1_r2(x[:, -1])
        return torch.cat([xy_coords, r1_r2], dim=1)