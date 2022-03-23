import torch.nn as nn
import torch
from DownSample import DownSample
from UpBlock import UpBlock
from double_conv import DoubleConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.bottle_neck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.downsample = DownSample(in_channels)
        self.ups = nn.ModuleList()
        self.final = nn.ConvTranspose2d(64, out_channels, kernel_size=(2, 2), stride=(2, 2))
        for feature in features[::-1]:
            self.ups.append(
                UpBlock(feature * 2, feature)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        bottle_input, downsample_output = self.downsample(x)
        out = self.bottle_neck(bottle_input)
        out = self.pool(out)
        j = len(downsample_output) - 1

        for i in range(0, len(self.ups), 2):
            out1 = self.ups[i](out)
            out2 = downsample_output[j]
            final = torch.cat([out1, out2], dim=1)
            out = self.ups[i+1](final)
            j -= 2

        out = self.final(out)
        return out
