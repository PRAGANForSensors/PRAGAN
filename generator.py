#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from pdb import set_trace as stx
# from torchsummary import summary
# from torchstat import stat
# from thop import profile

from ResizeNet import *
from lstmCell import *
from CBAM import *

cha_begin = 32

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Sequential(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride))

class Encoder(nn.Module):  # U-net Encoder
    def __init__(self):
        super(Encoder, self).__init__()
        # self.num = res_num

        self.Res1 = nn.Sequential(
                                  CBAM(cha_begin, 3),
                                  CBAM(cha_begin, 3),
                                  CBAM(cha_begin, 3))

        self.Res2 = nn.Sequential(CBAM(cha_begin * 2, 3),
                                  CBAM(cha_begin * 2, 3),
                                  CBAM(cha_begin * 2, 3))

        self.Res3 = nn.Sequential(CBAM(cha_begin * 4, 3),
                                  CBAM(cha_begin * 4, 3),
                                  CBAM(cha_begin * 4, 3))

        self.down1 = DownSamplePool(cha_begin, cha_begin)
        self.down2 = DownSamplePool(cha_begin * 2, cha_begin * 2)

        self.conv11 = CBAM(cha_begin, 1)
        self.conv22 = CBAM(cha_begin * 2, 1)

    def forward(self, inputs):

        x = self.Res1(inputs)
        out1 = self.conv11(x)
        x = self.down1(x)

        x = self.Res2(x)
        out2 = self.conv22(x)
        x = self.down2(x)

        x = self.Res3(x)

        return out1, out2, x


class Decoder(nn.Module):  # U-net Decoder
    def __init__(self):
        super(Decoder, self).__init__()

        self.Res1 = nn.Sequential(CBAM(cha_begin * 4, 3),
                                  CBAM(cha_begin * 4, 3),
                                  CBAM(cha_begin * 4, 3))

        self.Res2 = nn.Sequential(CBAM(cha_begin * 2, 3),
                                  CBAM(cha_begin * 2, 3),
                                  CBAM(cha_begin * 2, 3))

        self.Res3 = nn.Sequential(CBAM(cha_begin, 3),
                                  CBAM(cha_begin, 3),
                                  CBAM(cha_begin, 3),
                                  conv(cha_begin, 3, 3)
                                  )

        self.up1 = UpSample(cha_begin*2, cha_begin*2)
        self.up2 = UpSample(cha_begin, cha_begin)

    def forward(self, out1, out2, x):

        x = self.Res1(x)
        x = self.up1(x)
        x = x + out2

        x = self.Res2(x)
        x = self.up2(x)
        x = x + out1

        out = self.Res3(x)

        return out


class MTRUV(nn.Module):
    def __init__(self,channels, drop=False):
        super(MTRUV, self).__init__()
        self.drop = drop
        self.head = nn.Sequential(conv(channels, cha_begin, 3),
                                  CBAM(cha_begin, 3))
        self.E = Encoder()
        self.D = Decoder()

        self.cell = LSTMCell(cha_begin * 4, cha_begin, 3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, hidden=None, cell=None):

        x = self.head(input)
        out1, out2, x = self.E(x)
        x, hidden, cell = self.cell(x, hidden, cell)
        x = self.D(out1, out2, x)
        if self.drop:
            out = input - x
        else:
            out = input + x
        return torch.clamp(out, -1, 1), hidden, cell

