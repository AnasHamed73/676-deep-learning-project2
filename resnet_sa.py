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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNetSA:

    def __init__(self):
        self.model = _Model()
        self.criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.0, 0.9))
        # Decay LR by a factor of 0.1 every 7 epochs
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
   
    def loss_acc(self, dataloader, phase):
        # Iterate over data.
        dataset_size = len(dataloader.dataset)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            self.optimizer.zero_grad()
    
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
    
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
    
            # statistics
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
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
    
                epoch_loss, epoch_acc = self.loss_acc(dataloaders[phase], phase)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), '%s/best_model.pth' % '.')
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best validation Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     # fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['validation']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


class _Model(nn.Module):

    def __init__(self):
        super(_Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.model = self.model.to(device)
        # print(model_ft)
        self.blocks = list(self.model.children())

    def _forward(self, layer, input):
        print("layer: ", layer)
        print("input size: ", input.size())
        output = layer.forward(input)
        print("output size: ", output.size())
        print("________________________________")
        return output

    def forward(self, input):
        output = None
        for i, b in enumerate(self.blocks, 0):
            print("index: ", i)
            ip = output
            if i == 0:
                ip = input
            elif i == len(self.blocks) - 1:
                ip = torch.flatten(ip, 1)
            output = self._forward(b, ip)
        return output


# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=num_epochs)
#
# test_loss, test_acc = loss_acc(model_ft, dataloaders['test'], 'test', criterion, optimizer_ft, exp_lr_scheduler)
# print("test loss: {}, test acc: {}".format(test_loss, test_acc))
