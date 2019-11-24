#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

######MAIN

train_loss_file = "output_resnext_sa/train_loss"
val_loss_file = "output_resnext_sa/validation_loss"
output_img = "output_resnext_sa/resnext_sa_loss.png"

x_label = "Epoch"
y_label = "Loss"
#y_label = "Accuracy"
legend = ["Training Loss", "Validation Loss"]
#legend = ["Training Accuracy", "Validation Accuracy"]
title = "ResNextSA Loss"
epochs = 99

def get_losses(loss_file):
    loss = []
    with open(loss_file) as f:
        line = f.readline()
        while line != "":
            loss.append(float(line))
            line = f.readline()
    return loss

train_loss = get_losses(train_loss_file)
val_loss = get_losses(val_loss_file)
fig = plt.figure()
ax = plt.axes()
#x = np.linspace(0, 1000)
x = [i for i in range(epochs)] 

ax.plot(x, train_loss, 'r')
ax.plot(x, val_loss, 'b')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend(legend)
plt.title(title)
plt.savefig(output_img)

plt.show()
