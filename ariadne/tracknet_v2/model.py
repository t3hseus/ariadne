import gin
import torch
import torch.nn as nn


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

    def forward(self, inputs, input_lengths, return_gru_state=False, *args):
        # BxTxC -> BxCxT
        inputs = inputs.transpose(1, 2).float()
        input_lengths = input_lengths.int().cpu()
        ordered_lengths, ord_index = torch.sort(input_lengths, descending=True)
        ord_index = ord_index.to(inputs.device)
        ordered_inputs = torch.index_select(inputs, 0, ord_index)
        assert torch.all(inputs == torch.index_select(ordered_inputs, 0, ord_index.argsort(0))), 'Wrong sorting'
        if self.input_features > inputs.shape[1]:
            temp = torch.zeros(ordered_inputs.shape[0], 1, ordered_inputs.shape[2], device=ordered_inputs.device)
            temp[:, 0, :-1] = ordered_inputs[:, 0, 1:]
            temp[:, 0, -1] = 1.
            new_inputs = torch.cat((ordered_inputs, temp), 1)
            x = self.conv(new_inputs)
        else:
            x = self.conv(ordered_inputs)
        # BxCxT -> BxTxC
        x = x.transpose(1, 2)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, ordered_lengths, enforce_sorted=True, batch_first=True)
        # forward pass trough rnn
        x, _ = self.rnn(packed)
        # unpack padding
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = torch.index_select(x, 0, ord_index.argsort(0))
        last_nonzero = torch.index_select(ordered_lengths, 0, ord_index.argsort(0).to(ordered_lengths.device)).to(torch.long).to(x.device) - 1
        assert torch.all(input_lengths == torch.index_select(ordered_lengths, 0, ord_index.argsort(0).to(ordered_lengths.device))), 'Wrong unsorting of lengths'
       # mask = torch.not_equal(x, 0)
        #result = torch.masked_select(x, mask)
        #result = result.view(x.size(0), -1, x.size(2))
        #print(result[:2, -1])
        last_nonzero = last_nonzero.unsqueeze(-1)
        indices = last_nonzero.repeat(1, x.shape[-1]).unsqueeze(1)
        x = x.gather(1, indices)
        #print(x[:2])
        last_gru_output = x
        # get result using only the output on the last timestep
        xy_coords = self.xy_coords(x)
        r1_r2 = self.r1_r2(x)
        if return_gru_state:
            return torch.squeeze(torch.cat([xy_coords, r1_r2], dim=-1), dim=1), last_gru_output
        else:
            return torch.squeeze(torch.cat([xy_coords, r1_r2], dim=-1), dim=1)
