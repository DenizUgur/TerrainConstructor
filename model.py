import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                8,
                16,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                32,
                16,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                8,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class FingerNet(nn.Module):
    def __init__(self):
        super(FingerNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.conv10 = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        l1 = self.conv1(x)
        x = self.pool1(l1)

        l2 = self.conv2(x)
        x = self.pool2(l2)

        l3 = self.conv3(x)
        x = self.pool3(l3)

        l4 = self.conv4(x)
        x = self.pool4(l4)

        l5 = self.conv5(x)

        x = torch.cat((self.up1(l5), l4), dim=1)
        l6 = self.conv6(x)

        x = torch.cat((self.up2(l6), l3), dim=1)
        l7 = self.conv7(x)

        x = torch.cat((self.up3(l7), l2), dim=1)
        l8 = self.conv8(x)

        x = torch.cat((self.up4(l8), l1), dim=1)
        l9 = self.conv9(x)

        x = self.conv10(l9)
        return x


class YuNet(nn.Module):
    def __init__(self):
        super(YuNet, self).__init__()

        # * Encoding
        self.conv1 = PartialConv2d(
            in_channels=1, out_channels=64, kernel_size=7, padding=3
        )
        self.pool1 = nn.BatchNorm2d(64)

        self.conv2 = PartialConv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.pool2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = PartialConv2d(
            in_channels=128, out_channels=256, kernel_size=5, padding=2
        )
        self.pool3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv4 = PartialConv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.pool4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv5 = PartialConv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pool5 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv6 = PartialConv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pool6 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv7 = PartialConv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pool7 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv8 = PartialConv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.pool8 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # * Encoding Finished

        # * Decoding

    def forward(self, x):
        l1 = self.conv1(x)
        x = self.pool1(l1)

        l2 = self.conv2(x)
        x = self.pool2(l2)

        l3 = self.conv3(x)
        x = self.pool3(l3)

        l4 = self.conv4(x)
        x = self.pool4(l4)

        l5 = self.conv5(x)
        x = self.pool5(l5)

        l6 = self.conv6(x)
        x = self.pool6(l6)

        l7 = self.conv7(x)
        x = self.pool7(l7)

        l8 = self.conv8(x)
        x = self.pool8(l8)

        print(x.shape)

        # x = torch.cat((self.up1(l5), l4), dim=1)
        # l6 = self.conv6(x)

        # x = torch.cat((self.up2(l6), l3), dim=1)
        # l7 = self.conv7(x)

        # x = torch.cat((self.up3(l7), l2), dim=1)
        # l8 = self.conv8(x)

        # x = torch.cat((self.up4(l8), l1), dim=1)
        # l9 = self.conv9(x)

        # x = self.conv10(l9)
        return x


class CompletionNet(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # input_shape: (None, 1, img_h, img_w)
        self.conv0 = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2)
        self.bn0 = nn.BatchNorm2d(4)
        self.act0 = nn.ReLU()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, dilation=16, padding=16
        )
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(4)
        self.act17 = nn.ReLU()
        # output_shape: (None, 4, img_h. img_w)
        self.conv18 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.act18 = nn.Sigmoid()

    def forward(self, x):
        x = self.bn0(self.act0(self.conv0(x)))
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.bn17(self.act17(self.conv17(x)))
        x = self.act18(self.conv18(x))
        return x


class GeneratorNet(nn.Module):

    # generator model
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t6 = nn.Sequential(
            nn.Conv2d(512, 4000, kernel_size=(4, 4)), nn.BatchNorm2d(4000), nn.ReLU()
        )
        self.t7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.t8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.t9 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.t10 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=1, kernel_size=(4, 4), stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        x = self.t7(x)
        x = self.t8(x)
        x = self.t9(x)
        x = self.t10(x)
        return x  # output of generator


class DiscriminatorNet(nn.Module):

    # discriminator model
    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1, padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        return x  # output of discriminator


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv1x1(in_channels, out_channels),
        )


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"
    ):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(
        self,
        num_classes,
        in_channels=3,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="concat",
    ):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'upsampling. Only "transpose" and '
                '"upsample" are allowed.'.format(up_mode)
            )

        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for'
                "merging up and down paths. "
                'Only "concat" and '
                '"add" are allowed.'.format(up_mode)
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x