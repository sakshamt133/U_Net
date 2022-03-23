import torch.nn as nn
from double_conv import DoubleConv


class DownSample(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512], kernel_size=(3, 3)):
        super(DownSample, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature, kernel_size))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            in_channels = feature

    def forward(self, x):
        layers_out = []
        for layer in self.layers:
            x = layer(x)
            layers_out.append(x)

        return x, layers_out
