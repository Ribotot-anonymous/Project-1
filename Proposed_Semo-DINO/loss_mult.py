import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
import time, pdb
import numpy as np
from dino_utils import trunc_normal_

# This function is modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class LossFunction(nn.Module):
    def __init__(self, nOut, out_dim, use_bn_in_head=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, label=1211, **kwargs):
        super(LossFunction, self).__init__()

        self.last_layer = nn.utils.weight_norm(nn.Linear(nOut, label, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        nlayers = max(nlayers, 0)
        if nlayers == 0:
            self.mlp = nn.Identity()
        elif nlayers == 1:
            self.mlp = nn.Linear(nOut, nOut)
        else:
            layers = [nn.Linear(nOut, hidden_dim)]
            if use_bn_in_head:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn_in_head:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, nOut))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.dino_last_layer = nn.utils.weight_norm(nn.Linear(nOut, out_dim, bias=False))
        self.dino_last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.dino_last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label_num, ncrops):
        x_shape = x.size()
        x_sub_un = x_shape[0]//ncrops
        x = x.reshape(ncrops, x_sub_un, x_shape[-1])
        x_un = x[:,label_num:,:]
        x_la = x[:,:label_num,:]

        x_un = x_un.reshape(ncrops*(x_sub_un-label_num), x_shape[-1])
        x_un = self.mlp(x_un)
        x_un = nn.functional.normalize(x_un, dim=-1, p=2)
        x_un = self.dino_last_layer(x_un)

        x_la = x_la.reshape(ncrops*(label_num), x_shape[-1])
        x_la = nn.functional.normalize(x_la, dim=-1, p=2)
        x_la = self.last_layer(x_la)

        return x_la, x_un