
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import MSencoder
from .decoder import Udecoder
from .diffusiondecoder import DUdecoder
from .diffusionencoder import DUencoder
from .vit_seg_configs import get_b16_config
from .vit_fuse_atten import fuseTransformer

class Diffusion_net(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(Diffusion_net, self).__init__()
        self.classes = classes
        self.encoder = MSencoder(classes, channels)
        self.decoder = Udecoder(classes, channels)
        self.trans = fuseTransformer(get_b16_config(), img_size=64, inchannel=512, outchannel=512, ychannel=512)    
        self.difencoder = DUencoder(3, 3)
        self.difdecoder = DUdecoder(3, 3)
        

    def forward(self, x, x_noisy, t):
        
        x1, x2, x3, x4 = self.encoder(x)
        dx1, dx2, dx3, dx4 = self.difencoder(x_noisy, t)
        fuse_teature =  self.trans(torch.cat((x4, dx4), dim=1), torch.cat((dx4, x4), dim=1))
        out1 = self.decoder(x1, x2, x3, fuse_teature)
        out2 = self.difdecoder(x1, x2, x3, fuse_teature)

        return out1, out2
