import torch 
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
class ZLoot(nn.Module):


    def __init__(self, n_stations, radius_channel=False):
        super(ZLoot, self).__init__()
        if n_stations < 3:
            raise ValueError("Only detector configuration "
                             "with 3 or more stations is supported")

        self.n_stations = n_stations
        self.radius_channel = radius_channel
        # 5 features - X, Y and 3*Z (for each station)
        # in_channels = self.n_stations + 5
        in_channels = self.n_stations + 4
        # add radius channel
        if self.radius_channel:
            in_channels += 1     
        self.inconv = InConv(in_channels, 32)
        self.down1 = DownScale(32, 64)
        self.down2 = DownScale(64, 128)
        self.down3 = DownScale(128, 256)
        self.down4 = DownScale(256, 256)
        self.vertex_coords = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(8192, 3),
            nn.Linear(8192, 1),
            )


    def forward(self, inputs):
        x1 = self.inconv(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # flatten vector
        return self.vertex_coords(x5)
  

class X2Conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(X2Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = X2Conv(in_ch, out_ch)


    def forward(self, x):
        x = self.conv(x)
        return x


class DownScale(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownScale, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            X2Conv(in_ch, out_ch))


    def forward(self, x):
        x = self.mpconv(x)
        return x