#!/usr/bin/env python

from __future__ import print_function, division

import sys
sys.path.insert(1, './nets')
import torch
import torchvision
from numpy.distutils.system_info import numarray_info
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
from resnet_sa import ResNetSA

# plt.ion()

data_path = "/home/kikuchio/Documents/courses/deep-learning/project2/samples"
#data_path = "/home/kikuchio/courses/dl/project2/samples"
train_data_path = '/home/kikuchio/courses/dl/project2/samples/train'
fake_train_data_path = '/home/kikuchio/courses/dl/project2/hymenoptera_data'
validation_data_path = '/home/kikuchio/courses/dl/project2/samples/validation'
test_data_path = '/home/kikuchio/courses/dl/project2/samples/test'

batch_size = 64
num_epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(5)  # pause a bit so that plots are updated
#
#
# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
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
# def show_sample(data_loader):
#     for batch_idx, (data, target) in enumerate(test_loader):
#         print(target)
#         plt.imshow(np.transpose(data[1].numpy(), (1, 2, 0)))
#         plt.imshow(np.transpose(data[2].numpy(), (1, 2, 0)))
#         plt.show()
#         break


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

# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

model = ResNetSA()
model_weights_path = "/home/kikuchio/Documents/courses/deep-learning/project2/output_res_sa/best_model_sa.pth"
model.model.load_state_dict(torch.load(model_weights_path))
print("model loaded")
print("F1 score: ", model.f_score(dataloaders['test']))


#model = model.train_epochs(dataloaders, num_epochs)
#test_loss, test_acc = model.loss_acc(dataloaders['test'], 'test')
#print("test loss: {}, test acc: {}".format(test_loss, test_acc))

