# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/3 0003 17:33
import torch
import torch.nn.functional as F


class Inception(torch.nn.Module):
    """实现GoogleNet中Inception模块"""
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)  # 分支链接


class ResidualBlock(torch.nn.Module):
    """ResNet中残差块实现"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  # 先求和后激活

