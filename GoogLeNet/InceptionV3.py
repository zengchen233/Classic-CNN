import torch
import torch.nn as nn


class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2D(in_channels, out_channels=64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2D(in_channels, 48, kernel_size=1),
            BasicConv2D(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2D(in_channels, out_channels=64, kernel_size=1),
            BasicConv2D(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            BasicConv2D(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2D(in_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]     # [batch_size, c, h, w]，在通道上做拼接
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2D(in_channels, out_channels=384, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2D(in_channels, out_channels=64, kernel_size=1),
            BasicConv2D(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            BasicConv2D(in_channels=96, out_channels=96, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3stack = self.branch3x3stack(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch_pool, branch3x3, branch3x3stack]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2D(in_channels, out_channels=192, kernel_size=1)

        c7 = channels_7x7

        self.branch7x7 = nn.Sequential(
            BasicConv2D(in_channels, c7, kernel_size=1),
            BasicConv2D(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2D(c7, out_channels=192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2D(in_channels, c7, kernel_size=1),
            BasicConv2D(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2D(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2D(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2D(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2D(in_channels, out_channels=192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7stack = self.branch7x7stack(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch_pool, branch1x1, branch7x7, branch7x7stack]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2D(in_channels, out_channels=192, kernel_size=1),
            BasicConv2D(in_channels=192, out_channels=320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2D(in_channels, out_channels=192, kernel_size=1),
            BasicConv2D(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2D(in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2D(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch_pool, branch3x3, branch7x7]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2D(in_channels, out_channels=320, kernel_size=1)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2D(in_channels, out_channels=192, kernel_size=1)
        )
