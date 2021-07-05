import gin
import torch
import torch.nn as nn
from ariadne.layers import CausalConv1d


ALLOWED_RNN_TYPES = ['GRU', 'LSTM']

@gin.configurable
class TrackNETv2(nn.Module):
    """Builds TrackNETv2 model

    # Arguments
        input_features: number of input features (channels)
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
    """
    def __init__(self,
                 input_features=4,
                 conv_features=32,
                 rnn_type='gru',
                 batch_first=True,
                 use_causalconv=False):
        super().__init__()
        self.input_features = input_features
        rnn_type = rnn_type.upper()
        if rnn_type not in ALLOWED_RNN_TYPES:
            raise ValueError(f'RNN type {rnn_type} is not supported. '
                             f'Choose one of {ALLOWED_RNN_TYPES}')
        _rnn_layer = getattr(nn, rnn_type)
        if use_causalconv:
            self.conv = nn.Sequential(
                CausalConv1d(input_features, conv_features,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(conv_features)
            )
        else:
            self.conv = None
        self.rnn = _rnn_layer(
            input_size=input_features,
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

    def forward(self, inputs, input_lengths, return_gru_states=False):
        input_lengths = input_lengths.int().cpu()
        x = inputs

        if self.conv:
            # BxTxC -> BxCxT
            inputs = inputs.transpose(1, 2).float()
            x = self.conv(inputs)
            # BxCxT -> BxTxC
            x = x.transpose(1, 2)

        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, enforce_sorted=False, batch_first=True)
        # forward pass trough rnn
        x, _ = self.rnn(packed)
        # unpack padding
        gru_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # get result using only the output on the last timestep
        xy_coords = self.xy_coords(gru_outs)
        r1_r2 = self.r1_r2(gru_outs)
        outputs = torch.cat([xy_coords, r1_r2], dim=-1)
        if return_gru_states:
            return outputs, gru_outs
        return outputs


@gin.configurable
class TrackNETv2BES(nn.Module):
    """Builds TrackNETv2 model

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
        self.input_features = input_features
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

    def forward(self, inputs, input_lengths, return_gru_states=False):
        # BxTxC -> BxCxT
        inputs = inputs.transpose(1, 2).float()
        input_lengths = input_lengths.int().cpu()
        x = self.conv(inputs)
        # BxCxT -> BxTxC
        x = x.transpose(1, 2)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, enforce_sorted=False, batch_first=True)
        # forward pass trough rnn
        x, _ = self.rnn(packed)
        # unpack padding
        gru_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # get result using only the output on the last timestep
        xy_coords = self.xy_coords(gru_outs)
        r1_r2 = self.r1_r2(gru_outs)
        outputs = torch.cat([xy_coords, r1_r2], dim=-1)
        if return_gru_states:
            return outputs, gru_outs
        return outputs