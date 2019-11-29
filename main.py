#!/usr/bin/env python

from __future__ import print_function, division

import sys
# Add the path to the nets directory
sys.path.insert(1, '/home/kikuchio/Documents/courses/deep-learning/project2/nets')
import torch
import torchvision
from numpy.distutils.system_info import numarray_info
from torchvision import models, datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import numpy as np

from resnet import ResNet
from resnet_sa import ResNetSA
from resnext import ResNext
from resnext_sa import ResNextSA

data_path = "/home/kikuchio/Documents/courses/deep-learning/project2/samples"

batch_size = 64
num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dataloaders():
    phases = ['train', 'validation', 'test']
    normalization = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.4709, 0.0499, 0.0323), (0.1922, 0.2514, 0.2790))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_transforms = {
        'train': normalization, 'validation': normalization, 'test': normalization
    }
    
    data_dir = data_path 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes
    
    print(dataloaders['train'].dataset)
    print(dataloaders['validation'].dataset)
    print("class names: ", class_names)
    return dataloaders, dataset_sizes


###### MAIN

dataloaders, dataset_sizes = load_dataloaders()

model = ResNet()
print("model loaded")

model = model.train_epochs(dataloaders, num_epochs)
test_loss, test_acc = model.loss_acc(dataloaders['test'], 'test')
print("test loss: {}, test acc: {}".format(test_loss, test_acc))
print("F1 score: ", model.f_score(dataloaders['test']))

