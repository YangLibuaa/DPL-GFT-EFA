
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .vit_seg_configs import get_b16_config
from .vit_fuse_atten import fuseTransformer

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)



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


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResEncoder, self).__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        residual = self.conv1x1(x)
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = self.relu(self.bn1(self.conv1(x)))
        out = h+time_emb
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)

        return out

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.device = 'cuda'
    def forward(self, time):
        
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device).double() * -embeddings).double()
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.float()

class DUencoder(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(DUencoder, self).__init__()
        self.n_classes = classes
        time_emb_dim = 32
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU())
        self.encoder1 = ResEncoder(channels, 32, time_emb_dim)
        self.encoder2 = ResEncoder(32, 64, time_emb_dim)
        self.encoder3 = ResEncoder(64, 128, time_emb_dim)
        self.encoder_bottom = ResEncoder(128, 256, time_emb_dim)
        self.downsample = downsample()
        
        #initialize_weights(self)

    def forward(self, x, t):
        t_embedding = self.time_mlp(t)

        enc_input = self.encoder1(x, t_embedding)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder2(down1, t_embedding)
        down2 = self.downsample(enc1)

        enc2 = self.encoder3(down2, t_embedding)
        down3 = self.downsample(enc2)

        input_feature = self.encoder_bottom(down3, t_embedding)
        
        return enc_input, enc1, enc2, input_feature
