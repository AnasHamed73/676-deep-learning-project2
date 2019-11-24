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
#import matplotlib.pyplot as plt
import time
import os
import copy
#import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


class ResNextSA:

    def __init__(self):
        self.model = _Model()
        self.criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.0, 0.9))
        # Decay LR by a factor of 0.1 every 7 epochs
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        print(self.model)


    def f_score(self, dataloader):
        self.model.eval()
        tp = fp = fn = tn = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                batch_size = labels.size()[0]
                print("batch size: ", batch_size)
                for i in range(batch_size):
                    pred = preds[i]
                    actual = labels[i]
                    if pred == 0 and actual == 0:
                        tn += 1
                    elif pred == 1 and actual == 0:
                        fp += 1
                    elif pred == 0 and actual == 1:
                        fn += 1
                    elif pred == 1 and actual == 1:
                        tp += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("dataset size: ", len(dataloader.dataset))
        print("sum: ", tp+fp+fn+tn)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

   
    def loss_acc(self, dataloader, phase=None):
        dataset_size = len(dataloader.dataset)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
    
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
    
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            self.lr_scheduler.step()
    
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        return epoch_loss, epoch_acc

    def train_epochs(self, dataloaders, num_epochs=100):
        since = time.time()
    
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+40, num_epochs - 1 + 41))
            print('-' * 10)
    
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()  
    
                epoch_loss, epoch_acc = self.loss_acc(dataloaders[phase], phase)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), '%s/best_model.pth' % '.')
            #    break
            #break
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best validation Acc: {:4f}'.format(best_acc))
    
        self.model.load_state_dict(best_model_wts)
        return self


class _Model(nn.Module):

    def __init__(self):
        super(_Model, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.blocks = list(self.model.children())
        self._freeze_params()
        self._add_attn_blocks()
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].to(device)
        self.model = self.model.to(device)


    def _insert_layer(self, index, layer):
        self.blocks.insert(index, layer)


    def _add_attn_blocks(self):
        self.blocks.insert(5, _SelfAttention(256))
        self.blocks.insert(7, _SelfAttention(512))


    def _freeze_params(self):
        self.model.eval()
        for block in self.blocks:
            block.eval()


    def _forward(self, layer, input):
        #print("layer: ", layer)
        #print("input size: ", input.size())
        output = layer.forward(input)
        #print("output size: ", output.size())
        #print("________________________________")
        return output

    def forward(self, input):
        output = None
        for i, b in enumerate(self.blocks, 0):
            #print("index: ", i)
            ip = output
            if i == 0:
                ip = input
            elif i == len(self.blocks) - 1:
                ip = torch.flatten(ip, 1)
            output = self._forward(b, ip)
        return output


class _SelfAttention(nn.Module):

    def __init__(self, in_dim, ngpu=1):
        super(_SelfAttention, self).__init__()
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
