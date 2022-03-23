import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2,  2)):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x):
        return self.block(x)
