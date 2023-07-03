import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
    PerceiverModel,
    PerceiverClassificationPostprocessor,
)

from ariadne.pct.utils import sample_and_group


class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


@gin.configurable()
class PCT(nn.Module):
    def __init__(self, dropout=0.1, n_points=512, output_channels=40):
        super(PCT, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = LocalEncoder(in_channels=128, out_channels=128)
        self.gather_local_1 = LocalEncoder(in_channels=256, out_channels=256)

        self.pt_last = PointTransformerLast()

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, n_points, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_points),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(n_points, n_points, bias=False)
        self.bn6 = nn.BatchNorm1d(n_points)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(n_points, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, mask=None):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(
            npoint=self.n_points, radius=0.15, nsample=32, xyz=xyz, points=x
        )
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(
            npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature
        )
        feature_1 = self.gather_local_1(new_feature)
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PointTransformerLast(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        # channels = int(n_points / 4)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SALayer(channels)
        self.sa2 = SALayer(channels)
        self.sa3 = SALayer(channels)
        self.sa4 = SALayer(channels)

    def forward(self, x, mask=None):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x, mask=mask)
        x2 = self.sa2(x1, mask=mask)
        x3 = self.sa3(x2, mask=mask)
        x4 = self.sa4(x3, mask=mask)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class SALayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        if mask is not None:
            mask = torch.bitwise_and(mask[:, :, None].bool(), mask[:, None, :].bool())
            energy = energy.masked_fill(mask == 0, -9e15)
        else:
            energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

    def _generate_square_subsequent_mask(self, sequence_mask):
        mask = (
                torch.triu(
                    torch.ones(sequence_mask, sequence_mask)
                ) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1,
                                                                              float(
                                                                                  0.0))
        return mask


class ProcessorExp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(3, int(config.d_latents / 4), kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            int(config.d_latents / 4),
            int(config.d_latents / 4),
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(int(config.d_latents / 4))
        self.bn2 = nn.BatchNorm1d(int(config.d_latents / 4))

        self.gather_local_0 = LocalEncoder(
            in_channels=int(config.d_latents / 2),
            out_channels=int(config.d_latents / 2),
        )
        self.gather_local_1 = LocalEncoder(
            in_channels=int(config.d_latents), out_channels=int(config.d_latents)
        )

    def forward(self, x, mask=None):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(
            npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x
        )
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(
            npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature
        )
        feature_1 = self.gather_local_1(new_feature)

        return feature_1, None, None

    @property
    def num_channels(self) -> int:
        return self.config.d_model


@gin.configurable
class HFClassifierPerceiver(nn.Module):
    """Perceiver model based on HF Perceiver."""

    def __init__(self, d_model=256, num_heads=2, d_latents=128, num_latents=128):
        super().__init__()

        config = PerceiverConfig(
            d_model=d_model,
            num_heads=num_heads,
            d_latents=d_latents,
            num_latents=num_latents,
            num_cross_attention_heads=num_heads,
            num_self_attention_heads=num_heads,
            max_position_embeddings=num_latents,
            num_labels=40,
        )
        self.config = config
        preprocessor = ProcessorExp(config)
        decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=config.d_latents,
            num_channels=config.d_latents,
            trainable_position_encoding_kwargs=dict(
                num_channels=config.d_latents, index_dims=512
            ),
            use_query_residual=True,
        )
        self.model = PerceiverModel(
            config=config,
            input_preprocessor=preprocessor,
            decoder=decoder,
            output_postprocessor=PerceiverClassificationPostprocessor(
                config=config, in_channels=config.d_latents
            ),
        )
        self.postprocessor = nn.Sequential(
            nn.Conv1d(config.d_latents, int(config.d_latents / 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(int(config.d_latents / 2), 1, kernel_size=1),
        )

        self.linear = nn.Linear(config.d_latents, 1)

    def forward(self, **inputs):
        outputs = self.model(
            inputs["x"],
            # attention_mask=inputs['mask'],
            return_dict=True,
            output_attentions=True,
        )
        # outputs = inputs['x']
        # return self.postprocessor(outputs.last_hidden_state)
        # self.postprocessor(outputs.logits.transpose(-1, 1)).transpose(-1, 1)
        return outputs.logits  # self.linear(outputs.logits)


@gin.configurable()
class PCTSegment(nn.Module):
    def __init__(
        self, dropout: float =  0.1,
            n_points: int = 512,
            input_channels: int = 3,
            output_channels: int = 1,
            initial_permute: bool = True,
            n_patches: int = 4
    ):
        super().__init__()
        self.n_points = n_points
        self.initial_permute = initial_permute
        channels = n_points // 4
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.decoder_global = nn.Sequential(
            nn.Linear(n_points, n_points, bias=False),
            nn.BatchNorm1d(n_points),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear2 = nn.Linear(n_points, channels)
        self.bn7 = nn.BatchNorm1d(channels)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(channels, output_channels)
        self.pt_last = PointTransformerLast(channels=channels)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, n_points, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_points),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.lbr1 = nn.Sequential(
            nn.Linear(n_points*2, 256),
            nn.BatchNorm1d(n_points),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.lbr2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(n_points),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear_last = nn.Linear(256, output_channels)

    def forward(self, x, mask=None):
        if self.initial_permute:
            x = x.permute(0, 2, 1)
        batch_size, _, n = x.size()  # B, D, N

        x = self.encoder(x)
        x = self.pt_last(x, mask=mask)
        # global feature
        y = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        y = self.decoder_global(y)
        y = y.unsqueeze(1).transpose(2, 1)  # add new dimension
        y = y.repeat(1, 1, self.n_points)
        # segmentation
        x = torch.cat((x, y), 1).permute(0, 2, 1)
        x = self.lbr1(x)
        x = self.lbr2(x)
        x = self.linear_last(x)
        return x

