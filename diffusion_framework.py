
import sys, json, os, argparse
from shutil import copyfile, rmtree
import cv2
import os
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize
#from get_loaders import dataloader

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    返回所传递的值列表vals中的特定索引，同时考虑到批处理维度。
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device='cuda'):
    """ 
    接收一个图像和一个时间步长作为输入，并 返回它的噪声版本
    """
    noise = (torch.randn_like(x_0)+1)/2
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    #均值+方差
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def forward_diffusion_sample_last(x_0, t, device='cuda'):
    """ 
    接收一个图像和一个时间步长作为输入，并 返回它的上一步版本
    """
    noise = (torch.randn_like(x_0)+1)/2
    t_minusone = t.detach().cpu().numpy()-1
    t_minusone[t_minusone==-1] = 0
    t_minusone = torch.tensor(t_minusone).to(device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_alphas_cumprod_t_minusone = get_index_from_list(sqrt_alphas_cumprod, t_minusone, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t_minusone = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t_minusone, x_0.shape
    )
    #均值+方差
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), sqrt_alphas_cumprod_t_minusone.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t_minusone.to(device) * noise.to(device)



T = 100
betas = linear_beta_schedule(timesteps=T)
# 预先计算闭合形式的不同项
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
