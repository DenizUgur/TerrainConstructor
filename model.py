import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.kernel_size = 7
        self.stride = 1
        self.padding = 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
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
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                8,
                1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out