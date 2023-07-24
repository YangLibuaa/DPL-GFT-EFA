
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .vit_seg_configs import get_b16_config
from .vit_fuse_atten import fuseTransformer
from functools import partial
import torch_dct as dct
from .sobel import Gedge_map
from .sobel import edge_conv2d128, edge_conv2d64
from .sobel import edge_conv2d256

nonlinearity = partial(F.relu, inplace=True)

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out
    

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class local_freattention(nn.Module):
    def __init__(self, channel):
        super(local_freattention, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel // 2, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel // 2, 1, kernel_size=1, dilation=1, padding=0)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.u1 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        self.u2 = torch.nn.Parameter(torch.ones((1,1), dtype = torch.float32))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.psi = nn.Sequential(
            nn.Conv2d(channel // 2, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.frecoder = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        b, c, H, W = x.size()
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(self.dilate1(x)))
        dilate3_out = nonlinearity(self.dilate3(self.dilate2(self.dilate1(x))))

        fea1 = dilate1_out
        fea2 = dilate2_out
        fea3 = dilate3_out

        fea = fea1+fea2+fea3

        edgemap = self.relu(Gedge_map(self.psi(fea))+self.psi(fea))

        x = x*edgemap
        b, c, _, _ = x.size()
        y = self.max_pool(self.frecoder(dct.dct_2d(input))).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)+x

class DUdecoder(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(DUdecoder, self).__init__()
        self.n_classes = classes
        self.Up4 = up_conv(ch_in=512, ch_out=128)
        self.l_at4 = local_freattention(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.l_at3 = local_freattention(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.l_at2 = local_freattention(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)
        self.final = nn.Conv2d(32, classes, kernel_size=1)
        #initialize_weights(self)

    def forward(self, dx1, dx2, dx3, fuse_feature):
        
             
        # Do decoder operations here

        d4 = self.Up4(fuse_feature)
        lt2 = self.l_at4(dx3)
        x3 = lt2
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        lt3 = self.l_at3(dx2)
        x2 = lt3
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        lt4 = self.l_at2(dx1)
        x1 = lt4
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        final = self.final(d2)
        return final
