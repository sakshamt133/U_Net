from u_net import UNet
import torch


u = UNet(3, 1)
img = torch.randn((1, 3, 256, 256))
o = u(img)
print(o.shape)
