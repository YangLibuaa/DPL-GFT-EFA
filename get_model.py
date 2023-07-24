import sys
from dpvs_model.difmodel import Diffusion_net as unet

import torch


def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'unet':
        model = unet(classes=1, channels=3)


    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model

