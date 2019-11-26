#!/usr/bin/env python
from __future__ import print_function, division

from builtins import super

import torch
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


class SelfAttention(nn.Module):

    def __init__(self, in_dim, ngpu=1):
        super(SelfAttention, self).__init__()
        self.ngpu = ngpu

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, input):
        batch_size, nc, width, height = input.size()
        q = self.query(input).view(batch_size, -1, width*height).permute(0, 2, 1)
        k = self.key(input).view(batch_size, -1, width*height)
        qk = torch.bmm(q, k)

        # calc attention map
        attn = self.softmax(qk)
        v = self.value(input).view(batch_size, -1, width*height)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, nc, width, height)

        # append input back to attention
        out = (self.gamma * out) + input
        return out

