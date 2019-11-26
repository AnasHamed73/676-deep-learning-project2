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
from model_base import ModelBase
from self_attn import SelfAttention

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


class ResNetSA(ModelBase):

    def __init__(self):
        model = _Model()
        self.model = _Model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.0, 0.9))
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        print(self.model)


class _Model(nn.Module):

    def __init__(self):
        super(_Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.blocks = list(self.model.children())
        self._freeze_params()
        self._add_attn_blocks()
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].to(device)
        self.model = nn.Sequential(*self.blocks)
        self.model = self.model.to(device)


    def _add_attn_blocks(self):
        self.blocks.insert(4, SelfAttention(64))
        self.blocks.insert(6, SelfAttention(256))


    def _freeze_params(self):
        self.model.eval()
        for block in self.blocks:
            block.eval()


    def _forward(self, layer, input):
        output = layer.forward(input)
        return output

    def forward(self, input):
        output = None
        for i, b in enumerate(self.blocks, 0):
            ip = output
            if i == 0:
                ip = input
            elif i == len(self.blocks) - 1:
                ip = torch.flatten(ip, 1)
            output = self._forward(b, ip)
        return output


