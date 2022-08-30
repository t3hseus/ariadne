import gin
import torch
import torch.nn as nn
import math
from ariadne.layers import CausalConv1d


ALLOWED_RNN_TYPES = ['GRU', 'LSTM']

@gin.configurable
class TrackNETv2RNN(nn.Module):
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
        # BxTxC -> BxCxT
        #inputs = inputs.transpose(1, 2).float()
        input_lengths = input_lengths.int().cpu()
        # x = self.conv(inputs)
        x = inputs
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
                 use_causalconv=False,
                 use_rnn=True):
        super().__init__()
        self.input_features = input_features
        rnn_type = rnn_type.upper()
        assert use_causalconv + use_rnn, 'you need to use at least rnn or convolution'
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
        if use_rnn:
            self.rnn = _rnn_layer(
                input_size=conv_features if use_causalconv else input_features,
                hidden_size=conv_features,
                num_layers=2,
                batch_first=batch_first
            )
        else:
            self.rnn = None
        # outputs
        self.xy_coords = nn.Sequential(
            nn.Linear(conv_features, 2)
        )
        self.r1_r2 = nn.Sequential(
            nn.Linear(conv_features, 2),
            nn.Softplus()
        )

    def forward(self, inputs, input_lengths, mask=None, return_gru_states=False, return_x=False):
        input_lengths = input_lengths.int().cpu()
        x = inputs

        """if mask is not None:
            print(mask)
            for station in range(1, mask.shape[1] - 1):
                if not mask[:, station].all():
                    with torch.no_grad():
                        #print(inputs)
                        #print(station)
                        #print(mask)
                        #print(self.forward(x[:, :station + 1], torch.full_like(input_lengths, station + 1)))
                        print(station)
                        print(mask[:, :station + 1])
                        print((~mask).nonzero())
                        print(x[:, station, :2][(~mask[:, station]).nonzero()])
                        #print(self.forward(x[:, :], torch.full_like(input_lengths[:], station))[:, :, :2])
                        #print(self.forward(x, input_lengths)[:, :, :2][:, station - 1][~mask[:, station]])
                        output = self.forward(x, input_lengths)[:, :, :2][:, station - 1][~mask[:, station]].unsqueeze(0)
                        print(output)

                        print(x[:, station, :2][(~mask[:, station]).nonzero()].shape)
                        print(output.shape)
                        x[:, station, :2][(~mask[:, station]).nonzero()] = output.contiguous()
                        print(x)"""
        if mask is not None:
            #print(mask)
            #print(x)
            for track_id in range(mask.shape[0]):
                track_mask = mask[track_id]
                if not track_mask.all():
                    #print(track_mask)
                    #print(x[track_id])
                    with torch.no_grad():
                        for station in range(1, track_mask.shape[0]):
                            if not track_mask[station]:
                                #print(x[track_id][station])
                                output = self.forward(x[track_id].unsqueeze(0), torch.tensor([station], dtype=torch.int64))[0][-1, :2]
                                #print(output)
                                #print(output.shape)
                                x[track_id][station][:2] = output
            #print(x)
        

        if return_x:
            return x
        
        if self.conv:
            # BxTxC -> BxCxT
            inputs = inputs.transpose(1, 2).float()
            x = self.conv(inputs)
            # BxCxT -> BxTxC
            x = x.transpose(1, 2)

        if self.rnn:
            # Pack padded batch of sequences for RNN module
            packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False, batch_first=True)
            # forward pass trough rnn
            x, _ = self.rnn(packed)
            # unpack padding
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # get result using only the output on the last timestep
        xy_coords = self.xy_coords(x)
        r1_r2 = self.r1_r2(x)
        outputs = torch.cat([xy_coords, r1_r2], dim=-1)
        if return_gru_states and self.rnn:
            return outputs, x

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

@gin.configurable
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 max_len: int = 9):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :].repeat(1,
                                                                                   token_embedding.shape[1],
                                                                                   1))

@gin.configurable
class TrackNETv2Attention(nn.Module):
    """Builds TrackNETv2 model

    # Arguments
        input_features: number of input features (channels)
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
    """
    def __init__(self,
                 input_features=4,
                 conv_features=32,
                 use_pos_encoder=True,
                 dropout=0.1,
                 n_encoders=1
                 ):
        super().__init__()
        self.input_features = input_features
        self.use_pos_encoder = use_pos_encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_features,
            nhead=3,
            dim_feedforward=conv_features,
            dropout=dropout
        )
        norm = nn.LayerNorm(self.input_features)
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=n_encoders, norm=norm)
        if self.use_pos_encoder:
            self.position_encoder = PositionalEncoding(d_model=self.input_features, dropout=dropout, max_len=9)
        # outputs
        self.xy_coords = nn.Linear(self.input_features, 2)
        self.r1_r2 = nn.Sequential(
            nn.Linear(self.input_features, 2),
            nn.Softplus()
        )

    def generate_square_subsequent_mask(self, max_len):
        mask = torch.triu(torch.ones(max_len, max_len), 1).bool()
        return mask

    def forward(self, inputs, input_lengths, return_gru_states=False):
        # BxTxC -> TxBxC
        inputs = inputs.transpose(0, 1).float()
        src_mask = self.generate_square_subsequent_mask(inputs.size(0)).to(inputs.device)
        if self.use_pos_encoder:
            inputs = self.pos_encoder(inputs)
        x = self.encoder_layer(inputs, src_mask)
        x = x.transpose(0, 1)
        xy_coords = self.xy_coords(x)
        r1_r2 = self.r1_r2(x)
        outputs = torch.cat([xy_coords, r1_r2], dim=-1)
        if return_gru_states:
            return outputs, x
        return outputs