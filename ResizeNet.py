#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class DownSamplePool(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSamplePool, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class DownSampleStride(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSampleStride, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=2, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x