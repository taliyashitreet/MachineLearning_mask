import os
import argparse

import joblib
import numpy as np
import pandas as pd

#####

# imports
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random

import tarfile
import io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch
from torchvision import datasets, models, transforms
import torchvision.transforms as tvtf
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from collections import namedtuple
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
from sklearn import metrics


# from model import model

class TrueDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.dir = img_dir
        self.image_files = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        self.transform = transform
        self.pil2tensor = ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.pil2tensor(Image.open(self.image_files[idx]))
        label = int(os.path.splitext(self.image_files[idx])[0][-1])
        if self.transform:
            image = self.transform(image)
        return image.float(), label


import tarfile

my_tar = tarfile.open('train.tar')
my_tar.extractall('./train')  # specify which folder to extract to
my_tar.close()

print('Loading dataset')
import os

img_dir = "train/train"
datasets_masks = TrueDataset(img_dir)

# find the mean and std of the images
mean = torch.tensor([0.5, 0.4, 0.4])
std = torch.tensor([0.2, 0.2, 0.2])

datasets_masks = TrueDataset(img_dir, transforms.Compose(
    [transforms.Resize((96, 96)), transforms.Normalize(mean=mean, std=std)]))

train_loader = DataLoader(datasets_masks, batch_size=32, shuffle=True)

# loading dataset
print('Loading test')

my_tar = tarfile.open('test.tar')
my_tar.extractall('./test')  # specify which folder to extract to
my_tar.close()

import os

img_dir = "test/test"
datasets_masks = TrueDataset(img_dir)

datasets_masks = TrueDataset(img_dir, transforms.Compose(
    [transforms.Resize((96, 96)), transforms.Normalize(mean=mean, std=std)]))

test_loader = DataLoader(datasets_masks, batch_size=100, shuffle=False)


# loading teat


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        self.dropout = nn.Dropout(0.2)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet18_config = ResNetConfig(block=BasicBlock,
                               n_blocks=[2, 2, 2, 2],
                               channels=[64, 128, 256, 512])
model = ResNet(resnet18_config, 2)

CE_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, test_losses = [], []
F1_train, F1_test = [], []
y_pred_train, y_pred_test, y_true_test, y_true_train = [], [], [], []
y_pred_train_label, y_pred_test_label = [], []

epochs = 13
for i in range(epochs):
    running_loss = 0
    # Loop through all of the train set forward and back propagate
    # train
    for images, labels in train_loader:
        optimizer.zero_grad()
        log_ps = model(images)
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(log_ps)
        #y_pred_train += y_pred
        ps = torch.exp(log_ps)
        loss = CE_loss(log_ps, labels)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        top_class1 = top_class.view(-1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Initialize test loss and accuracy to be 0

    test_loss = 0
    accuracy = 0
    total_examples = 0

    # Turn off the gradients
    # test
    with torch.no_grad():
        # Loop through all of the validation set
        for images, labels in test_loader:
            log_ps = model(images)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(log_ps)
            y_pred_test += y_pred
            ps = torch.exp(log_ps)
            test_loss += CE_loss(log_ps, labels)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            top_class1 = top_class.view(-1)
            y_pred_test_label += top_class1
            total_examples += images.shape[0]
            accuracy += (equals * 1).sum().item()
            y_true_test += labels

    # we want to calculate- model performance on train
    # train
    with torch.no_grad():
        for images, labels in train_loader:
            log_ps = model(images)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(log_ps)
            y_pred_train += y_pred
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            top_class1 = top_class.view(-1)
            y_pred_train_label += top_class1
            y_true_train += labels

        y_pred_train_prob = []
        for item in y_pred_train:
            y_pred_train_prob.append(float(item[1]))

        y_pred_test_prob = []
        for item in y_pred_test:
            y_pred_test_prob.append(float(item[1]))


        # Append the average losses to the array for plotting
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        tens_true_train = torch.tensor(y_true_train)
        tens_true_test = torch.tensor(y_true_test)


        print("F1 score train", f1_score(tens_true_train, y_pred_train_label))
        print("F1 score test", f1_score(tens_true_test, y_pred_test_label))
        F1_train.append(f1_score(tens_true_train, y_pred_train_label))
        F1_test.append(f1_score(tens_true_test, y_pred_test_label))

    print('Epoch: [{}/{}], Training Loss: {:.4}, Test Loss: {:.4}, Test Accuracy: {:.4}'.format(i + 1, epochs,
                                                                                                train_losses[-1],
                                                                                                test_losses[-1],
                                                                                                accuracy / total_examples))

np.save('train_loss.npy', train_losses)
np.save('test_losses.npy', test_losses)
np.save('F1_train.npy', F1_train)
np.save('F1_test.npy', F1_test)
np.save('y_true_test.npy', y_true_test)
np.save('y_true_train.npy', y_true_train)
np.save('y_pred_test_prob.npy', y_pred_test_prob)
np.save('y_pred_train_prob.npy', y_pred_train_prob)


torch.save(model, 'model.pt')
####
