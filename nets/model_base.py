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


# Some code borrowed from the offical PyTorch tutorial on Transfer learning:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
class ModelBase:

    def f_score(self, dataloader):
        self.model.eval()
        tp = fp = fn = tn = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                batch_size = labels[0].item()
                print("batch size: ", batch_size)
                for i in range(batch_size):
                    print("pred is {}, actual is {}".format(pred, actual))
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
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best validation Acc: {:4f}'.format(best_acc))
    
        self.model.load_state_dict(best_model_wts)
        return self


