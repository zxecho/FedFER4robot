import torch
from torch import nn
import torch.nn.functional as F


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=3, classes=1000):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        blocks.append(my_IncepResNet_unit(256, 1))
        for i in range(3):
            blocks.append(my_IncepResNet_unit(128, 1))
            blocks.append(nn.MaxPool2d(2, stride=1, padding=0))
        # blocks.append(Inception_ResNet_B(256, 1))
        # for i in range(2):
        # blocks.append(Inception_ResNet_C(256, 1))
        # blocks.append(Inception_ResNet_C(256, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(128, 1024, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class My_IncepRestNet(nn.Module):
    def __init__(self, in_channels=3, classes=6):
        super(My_IncepRestNet, self).__init__()

        self.stem = Stem(in_channels)
        self.incepres_1 = my_IncepResNet_unit(224, 64, 1)
        self.incepres_2 = my_IncepResNet_unit(64, 64, 1)
        self.incepres_3 = my_IncepResNet_unit(64, 128, 1)

        self.conv = Conv2d(128, 512, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incepres_1(x)
        x = self.incepres_2(x)
        x = self.incepres_3(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MySimpleNet(nn.Module):
    def __init__(self, in_channels=3, classes=6):
        super(MySimpleNet, self).__init__()

        self.stem = SimpleStem(in_channels)
        self.incepres_1 = my_IncepResNet_unit(160, 64, 1)
        self.incepres_2 = my_IncepResNet_unit(64, 64, 1)
        self.incepres_3 = my_IncepResNet_unit(64, 64, 1)

        self.conv = Conv2d(64, 256, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incepres_1(x)
        x = self.incepres_2(x)
        x = self.incepres_3(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=1, padding=0, bias=False),  # 149 x 149 x 32
            # Conv2d(32, 32, 3, stride=1, padding=0, bias=False),  # 147 x 147 x 32
            # Conv2d(32, 32, 3, stride=1, padding=1, bias=False),  # 147 x 147 x 64
            # nn.MaxPool2d(2, stride=1, padding=0),  # 73 x 73 x 64
        )

        self.branch_0 = Conv2d(32, 32, 1, stride=1, padding=0, bias=False)

        self.branch_1 = nn.Sequential(
            Conv2d(32, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(32, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False),
            Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(32, 64, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class SimpleStem(nn.Module):
    def __init__(self, in_channels):
        super(SimpleStem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=1, padding=0, bias=False),  # 149 x 149 x 32
            # Conv2d(32, 32, 3, stride=1, padding=0, bias=False),  # 147 x 147 x 32
            # Conv2d(32, 32, 3, stride=1, padding=1, bias=False),  # 147 x 147 x 64
            # nn.MaxPool2d(2, stride=1, padding=0),  # 73 x 73 x 64
        )

        self.branch_0 = Conv2d(32, 32, 1, stride=1, padding=0, bias=False)

        self.branch_1 = nn.Sequential(
            Conv2d(32, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(32, 32, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class my_IncepResNet_unit(nn.Module):
    def __init__(self, in_channels, br_channel=32, scale=1.0):
        super(my_IncepResNet_unit, self).__init__()
        self.scale = scale

        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)

        # self.branch_1 = nn.Sequential(
        #     Conv2d(in_channels, br_channel, 1, stride=1, padding=0, bias=False),
        #     Conv2d(br_channel, br_channel*2, 3, stride=1, padding=1, bias=False)
        # )

        self.branch_2 = nn.Sequential(
            # Conv2d(in_channels, br_channel, 1, stride=1, padding=0, bias=False),
            Conv2d(in_channels, br_channel, 3, stride=1, padding=1, bias=False),
            Conv2d(br_channel, br_channel, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, br_channel, 3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x = self.maxpool(x)
        # x0 = self.branch_0(x)
        # x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        # x_res = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        return self.prelu(x + self.scale * x2)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 64, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 64, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            Conv2d(64, 128, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv2d(128, in_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 128, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(128, 128, (7, 1), stride=1, padding=(3, 0), bias=False)
        )
        self.conv = nn.Conv2d(256, in_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 64, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(64, 128, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(160, in_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.gn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.gn(x)
        x = self.relu(x)
        return x
