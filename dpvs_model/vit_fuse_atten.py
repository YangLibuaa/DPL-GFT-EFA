# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.autograd import Variable
from scipy import ndimage


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(conv, self).__init__()
        self.cov = nn.Sequential(nn.Conv2d(in_channels = inchannel, out_channels = outchannel, kernel_size = 3, stride = 1, padding=1),
                                 nn.BatchNorm2d(outchannel),
                                 nn.ReLU(inplace=True)
                                 )
    def forward(self,x):
        #x = self.down(x)
        x = self.cov(x)
        return x
        

class fuseAttention(nn.Module):
    def __init__(self, config, vis, inchannel, mode='w'):
        super(fuseAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = inchannel
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.mode = mode

        self.query1 = Linear(self.all_head_size, self.all_head_size)
        self.key1 = Linear(self.all_head_size, self.all_head_size)
        self.value1 = Linear(self.all_head_size, self.all_head_size)

        self.query2 = Linear(self.all_head_size, self.all_head_size)
        self.key2 = Linear(self.all_head_size, self.all_head_size)
        self.value2 = Linear(self.all_head_size, self.all_head_size)

        self.gate_conv = Conv2d(in_channels=inchannel,out_channels=self.all_head_size,kernel_size=3,padding=1)
        self.gate_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gate = Linear(self.all_head_size, self.all_head_size)

        self.out = Linear(self.all_head_size, self.all_head_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input1, input2):
        mixed_query_layer1 = self.query1(input1)
        mixed_key_layer1 = self.key1(input1)
        mixed_value_layer1 = self.value1(input1)

        mixed_query_layer2 = self.query2(input2)
        mixed_key_layer2 = self.key2(input2)
        mixed_value_layer2 = self.value2(input2)

        mixed_key_layer1 = mixed_key_layer1+mixed_key_layer1*torch.sigmoid(mixed_key_layer2) #the gated attention
        mixed_key_layer2 = mixed_key_layer2+mixed_key_layer2*torch.sigmoid(mixed_key_layer1) #the gated attention

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        attention_scores1 = torch.matmul(query_layer1, key_layer1.transpose(-1, -2))
        attention_scores2 = torch.matmul(query_layer2, key_layer2.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_probs1 = self.softmax(attention_scores1)
        attention_probs2 = self.softmax(attention_scores2)

        weights1 = attention_probs1 if self.vis else None
        weights2 = attention_probs2 if self.vis else None

        attention_probs1 = self.attn_dropout(attention_probs1)
        attention_probs2 = self.attn_dropout(attention_probs2)
        #print(attention_probs.shape)
        
        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)

        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer1.size()[:-2] + (self.all_head_size,)
        #print(context_layer.shape)
        context_layer1 = context_layer1.view(*new_context_layer_shape)
        context_layer2 = context_layer2.view(*new_context_layer_shape)

        attention_output1 = self.out(context_layer1)
        attention_output1 = self.proj_dropout(attention_output1)

        attention_output2 = self.out(context_layer2)
        attention_output2 = self.proj_dropout(attention_output2)

        attention_output = torch.cat((attention_output1, attention_output2), dim=2)

        return attention_output, weights1


class fuseMlp(nn.Module):
    def __init__(self, config, inchannel):
        super(fuseMlp, self).__init__()
        self.numhead = config.transformer["num_heads"]
        self.fc1 = Linear(self.numhead*inchannel*2, inchannel)
        self.fc2 = Linear(inchannel, self.numhead*inchannel)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

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


class fuseEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, imsize, inchannel):
        super(fuseEmbeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = (imsize, imsize)
        #patch_size = (imsize//64, imsize//64)
        patch_size = (2, 2)
        #patch_sizex = (1, 1)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=inchannel,
                                       out_channels=inchannel*config.transformer["num_heads"],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.weth_position_embeddings = nn.Parameter(torch.zeros(1, inchannel*config.transformer["num_heads"], img_size[0] // patch_size[0], 1))

        self.heth_position_embeddings = nn.Parameter(torch.zeros(1, inchannel*config.transformer["num_heads"], 1, img_size[0] // patch_size[0]))
        
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, inchannel*config.transformer["num_heads"]))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        
        x = x+torch.matmul(self.weth_position_embeddings,self.heth_position_embeddings)
        x = x.flatten(2)
        
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        
        embeddings = x
        embeddings = self.dropout(embeddings)
        return embeddings


class fuseBlock(nn.Module):
    def __init__(self, config, vis, inchannel):
        super(fuseBlock, self).__init__()
        self.hidden_size = inchannel*config.transformer["num_heads"]
        self.attention_norm1 = LayerNorm(self.hidden_size, eps=1e-6)
        self.attention_norm2 = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size*2, eps=1e-6)
        self.ffn = fuseMlp(config, inchannel)
        self.attn = fuseAttention(config, vis, inchannel)
        
    def forward(self, input1, input2):
        
        out1 = self.attention_norm1(input1)
        out2 = self.attention_norm2(input2)
        x, weights = self.attn(out1, out2)

        h = out1
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class fuseEncoder(nn.Module):
    def __init__(self, config, vis, inchannel):
        super(fuseEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(inchannel*config.transformer["num_heads"], eps=1e-6)
        for _ in range(1):
            layer = fuseBlock(config, vis, inchannel)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, input1, input2):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(input1, input2)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer_fuse(nn.Module):
    def __init__(self, config, imsize, vis, inchannel):
        super(Transformer_fuse, self).__init__()
        self.embedding1 = fuseEmbeddings(config, imsize, inchannel)
        self.embedding2 = fuseEmbeddings(config, imsize, inchannel)
        self.encoder = fuseEncoder(config, vis, inchannel)

    def forward(self, input1, input2):
        output1 = self.embedding1(input1)
        output2 = self.embedding1(input2)

        encoded, attn_weights = self.encoder(output1, output2)  # (B, n_patch, hidden)
        return encoded, attn_weights


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class fuseDecoder(nn.Module):
    def __init__(self, config, inchannel, outchannel):
        super().__init__()
        self.config = config
        head_channels = outchannel
        self.conv_more = Conv2dReLU(
            inchannel*config.transformer["num_heads"],
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.upsam = up_conv(ch_in=outchannel, ch_out=outchannel)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.upsam(x)
        return x

class fuseTransformer(nn.Module):
    def __init__(self, config, img_size=128, vis=False, inchannel=256, outchannel=256, ychannel=128):
        super(fuseTransformer, self).__init__()
        self.transformer = Transformer_fuse(config, img_size, vis, inchannel)
        self.decoder = fuseDecoder(config, inchannel, outchannel)
        self.config = config
        self.cov = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1)
    def forward(self, x, y):
        input = x
        out, attn_weights = self.transformer(x, y)  # (B, n_patch, hidden)

        out = self.decoder(out)
        out = out+self.cov(input)
        return out
