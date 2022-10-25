import math

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    based on the basic ResNet Block used in torchvision
    inspired on https://jarvislabs.ai/blogs/resnet
    """

    def __init__(self, n_channels_in, n_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, n_channels)  # change to LN to avoid batch size dependency (num rotations..)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, n_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class MaxPoolDownSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()
        padding = math.floor(kernel_size / 2)
        self.conv = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            stride=1,  # striding is a cheap way to downsample, but it is less informative that Pooling after full conv.
            dilation=1,  # dilation is a cheap way to increase receptive field, but it is less informative than deeper networks or downsampling..
            padding=padding,
            bias=True,
        )
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=n_channels_in * 2,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=True,
            padding="same",
        )
        self.relu = nn.ReLU()

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        x = self.relu(x)

        return x


class Unet(nn.Module):
    def __init__(
        self, n_channels_in=3, n_downsampling_layers=2, n_resnet_blocks=3, n_channels=32, kernel_size=3, **kwargs
    ):
        super().__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size, padding="same")

        # create ModuleLists to ensure layers are discoverable by torch (lightning) for e.g. model summary and bringing to cuda.
        # https://pytorch.org/docs/master/generated/torch.nn.ModuleList.html#torch.nn.ModuleList
        self.downsampling_blocks = nn.ModuleList(
            [MaxPoolDownSamplingBlock(n_channels, n_channels, kernel_size) for _ in range(n_downsampling_layers)]
        )
        self.resnet_blocks = nn.ModuleList([ResNetBlock(n_channels, n_channels) for _ in range(n_resnet_blocks)])
        self.upsampling_blocks = nn.ModuleList(
            [
                UpSamplingBlock(n_channels_in=n_channels, n_channels_out=n_channels, kernel_size=kernel_size)
                for _ in range(n_downsampling_layers)
            ]
        )

    def forward(self, x):
        skips = []

        x = self.conv1(x)

        for block in self.downsampling_blocks:
            skips.append(x)
            x = block(x)

        for block in self.resnet_blocks:
            x = block(x)

        for block in self.upsampling_blocks:
            x_skip = skips.pop()
            x = block(x, x_skip)
        return x

    def get_n_channels_out(self):
        return self.n_channels


if __name__ == "__main__":
    u = Unet(4)
    x = torch.randn(4, 4, 64, 64)
    y = u(x)
    print(y.shape)
