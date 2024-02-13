import gin
import torch
import torch.nn as nn
import torch.nn.functional as f


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
        batch_size, _, _ = x.size()

        # B, D, N
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
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

    @staticmethod
    def _generate_square_subsequent_mask(sequence_mask):
        mask = (torch.triu(torch.ones(sequence_mask, sequence_mask)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


@gin.configurable()
class PCTSegment(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        n_points: int = 512,
        input_channels: int = 3,
        output_channels: int = 1,
        initial_permute: bool = True,
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
            nn.Linear(n_points * 2, 256),
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
        y = f.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        y = self.decoder_global(y)
        y = y.unsqueeze(1).transpose(2, 1)  # add new dimension
        y = y.repeat(1, 1, self.n_points)
        # segmentation
        x = torch.cat((x, y), 1).permute(0, 2, 1)
        x = self.lbr1(x)
        x = self.lbr2(x)
        x = self.linear_last(x)
        return x
