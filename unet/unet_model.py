""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 128, 64)  # 4
        self.down1 = Down(128, 128, pool=False)  # 5
        self.down2 = Down(128, 256)  # 6
        self.down3 = Down(256, 256, pool=False)  # 7
        self.down4 = Down(256, 512)  # 8
        self.down5 = Down(512, 512, pool=False)  # 9
        self.down6 = Down(512, 512)  # 10
        self.down7 = Down(512, 512, pool=False)  # 11
        self.down8 = Down(512, 512)  # 12
        self.down9 = Down(512, 512, pool=False)  # 13
        self.down10 = Down(512, 512)  # 14
        self.down11 = Down(512, 512, pool=False)  # 15
        self.down12 = Down(512, 512)  # 16
        self.down13 = Down(512, 512, pool=False)  # 17

        self.down14 = Down(512, 512)  # 18 2x2 (index restart)

        self.up = Up(512, 512)  # 1, 3, 5, 7, 9, 11, 13

        self.h1 = Down(1024, 512, pool=False)  # 2, 4, 6, 8
        self.h2 = Down(768, 256, pool=False)  # 10
        self.h3 = Down(384, 128, pool=False)  # 12
        self.h4 = Down(192, 64, pool=False)  # 14
        self.outc = OutConv(64, n_classes)  # 15

    def forward(self, x):
        x0, x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x10 = self.down9(x9)
        x11 = self.down10(x10)
        x12 = self.down11(x11)
        x13 = self.down12(x12)
        x14 = self.down13(x13)
        x15 = self.down14(x14)  # 0

        u1 = self.up(x15, x13)  # 1
        u2 = self.h1(u1)  # 2

        u3 = self.up(u2, x11)  # 3
        u4 = self.h1(u3)  # 4

        u5 = self.up(u4, x9)  # 5
        u6 = self.h1(u5)  # 6

        u7 = self.up(u6, x7)  # 7
        u8 = self.h1(u7)  # 8

        u9 = self.up(u8, x3)  # 9
        u10 = self.h2(u9)  # 10

        u11 = self.up(u10, x1)  # 11
        u12 = self.h3(u11)  # 12

        u13 = self.up(u12, x0)  # 13
        u14 = self.h4(u13)  # 14

        return self.outc(u14)
