# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：ResNet.py
@Author ：AnthonyZ
@Date ：2022/6/2 15:36
"""


import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.batch_normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample=False):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.triplet_attention = TripletAttention(out_channels * 4)
        if self.down_sample:
            self.sample = DownSample(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            x = self.sample(x)
        out = self.triplet_attention(out)
        out = self.relu2(out + x)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.basic1 = BasicBlock(32, 32, 1)
        self.basic2 = BasicBlock(32, 64, 2, True)
        self.basic3 = BasicBlock(64, 64, 1)
        self.basic4 = BasicBlock(64, 128, 2, True)
        self.basic5 = BasicBlock(128, 128, 1)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.basic1(x)
        x = self.basic2(x)
        x = self.basic3(x)
        x = self.basic4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu2(x)
        return self.fc2(x)

