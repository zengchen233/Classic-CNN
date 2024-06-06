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

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]  # [batch_size, c, h, w]，在通道上做拼接
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

        self.branch3x3_1 = BasicConv2D(in_channels, out_channels=384, kernel_size=1)
        self.branch3x3_2a = BasicConv2D(in_channels=384, out_channels=384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2D(in_channels=384, out_channels=384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2D(in_channels, out_channels=448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2D(in_channels=448, out_channels=384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2D(in_channels=384, out_channels=384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2D(in_channels=384, out_channels=384, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch_pool = self.branch_pool(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        outputs = [branch_pool, branch3x3, branch3x3stack, branch1x1]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=100):
        super(InceptionV3, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2D(3, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2D(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2D(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2D(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2D(80, 192, kernel_size=3)

        # naive inception module
        self.Mixed_5b = InceptionA(in_channels=192, pool_features=32)
        self.Mixed_5c = InceptionA(in_channels=256, pool_features=64)
        self.Mixed_5d = InceptionA(in_channels=288, pool_features=64)

        # downsample
        self.Mixed_6a = InceptionB(in_channels=288)

        self.Mixed_6b = InceptionC(in_channels=768, channels_7x7=128)
        self.Mixed_6c = InceptionC(in_channels=768, channels_7x7=160)
        self.Mixed_6d = InceptionC(in_channels=768, channels_7x7=160)
        self.Mixed_6e = InceptionC(in_channels=768, channels_7x7=192)

        # downsample
        self.Mixed_7a = InceptionD(in_channels=768)

        self.Mixed_7b = InceptionE(in_channels=1280)
        self.Mixed_7c = InceptionE(in_channels=2048)

        # 6*6 feature size
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout2d()  # 按通道丢弃
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        # 32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        # 30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        # 30 -> 14
        # Efficient Grid Size Reduction to avoid representation
        # bottleneck
        x = self.Mixed_6a(x)

        # 14 -> 14
        # """In practice, we have found that employing this factorization does not
        # work well on early layers, but it gives very good results on medium
        # grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        # On that level, very good results can be achieved by using 1 × 7 convolutions
        # followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # 14 -> 6
        # Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        # 6 -> 6
        # We are using this solution only on the coarsest grid,
        # since that is the place where producing high dimensional
        # sparse representation is the most critical as the ratio of
        # local processing (by 1 × 1 convolutions) is increased compared
        # to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # 6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def inceptionV3(num_classes=100):
    return InceptionV3(num_classes=num_classes)
