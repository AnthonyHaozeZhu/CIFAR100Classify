# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：ResNet.py
@Author ：AnthonyZ
@Date ：2022/6/2 15:36
"""

import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, img_shape, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.attention = nn.MultiheadAttention(
            embed_dim=img_shape * 2,
            num_heads=1,
            batch_first=True)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        # print("x:", x.shape)
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        # print(y.shape)

        y = self.conv1(y)

        y = torch.squeeze(y)
        y = self.attention(y, y, y)[0]
        # print(y.shape)
        y = torch.unsqueeze(y, dim=-1)

        y = self.bn1(y)
        # print(y.shape)

        # print("y:", y.shape)
        y = self.act(y)
        # 将这一层的两个操作 替换成multi head self-attention
        # print("y2:", y.shape)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        # print("x_h", x_h.shape)
        x_w = x_w.permute(0, 1, 3, 2)
        # print("x_w", x_w.shape)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # print("a_w", a_w.shape, a_h.shape)
        # print("id", identity.shape)
        out = identity * a_w * a_h

        return out


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
    def __init__(self, in_channels, out_channels, stride, img_shape, down_sample=False):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.attention = CoordAtt(out_channels, out_channels, img_shape)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.down_sample:
            self.sample = DownSample(in_channels, out_channels)

    def forward(self, x):
        a = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.attention(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            a = self.sample(x)
        out = self.relu2(out + a)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.basic1 = BasicBlock(32, 32, 1, 32)
        self.basic2 = BasicBlock(32, 64, 2, 32, True)
        self.basic3 = BasicBlock(64, 64, 1, 16)
        self.basic4 = BasicBlock(64, 128, 2, 16, True)
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

