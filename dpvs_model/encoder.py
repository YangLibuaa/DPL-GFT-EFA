
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MSencoder(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(MSencoder, self).__init__()
        self.n_classes = classes
        self.encoder1 = conv_block(channels, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 128)
        self.encoder_bottom = conv_block(128, 256)
        self.downsample = downsample()
        
        #initialize_weights(self)

    def forward(self, x):
        
        enc_input = self.encoder1(x)
        
        down1 = self.downsample(enc_input)

        enc1 = self.encoder2(down1)
        
        down2 = self.downsample(enc1)

        enc2 = self.encoder3(down2)
        
        down3 = self.downsample(enc2)


        input_feature = self.encoder_bottom(down3)

        return enc_input, enc1, enc2, input_feature
