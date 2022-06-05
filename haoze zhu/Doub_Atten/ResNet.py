# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：ResNet.py
@Author ：AnthonyZ
@Date ：2022/6/2 15:36
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c):
        """
        :param
        in_c: 进行注意力refine的特征图的通道数目；
        原文中的降维和升维没有使用
        """
        super(DoubleAtten,self).__init__()
        self.in_c = in_c
        """
        以下对同一输入特征图进行卷积，产生三个尺度相同的特征图，即为文中提到A, B, V
        """
        self.convA = nn.Conv2d(in_c, in_c, kernel_size=(1, 1))
        self.convB = nn.Conv2d(in_c, in_c, kernel_size=(1, 1))
        self.convV = nn.Conv2d(in_c, in_c, kernel_size=(1, 1))

    def forward(self, input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w)  # 对 A 进行reshape
        atten_map = atten_map.view(b, self.in_c, 1, h*w)        # 对 B 进行reshape 生成 attention_aps
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)), dim=-1)  # 特征图与attention_maps 相乘生成全局特征描述子

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1)  # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0, 2, 1)  # 注意力向量左乘全局特征描述子

        return out.view(b, _, h, w)


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

