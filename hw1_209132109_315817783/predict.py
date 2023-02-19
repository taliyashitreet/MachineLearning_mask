import os
import argparse
from collections import namedtuple

import numpy as np
import pandas as pd
import joblib
import torch
from PIL import Image
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


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


def main():
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    # Reading input folder
    files = os.listdir(args.input_folder)

    #####
    model = torch.load('model.pt')
    ####


    mean = torch.tensor([0.5, 0.4, 0.4])
    std = torch.tensor([0.2, 0.2, 0.2])

    datasets_masks = TrueDataset(args.input_folder, transforms.Compose(
        [transforms.Resize((96, 96)), transforms.Normalize(mean=mean, std=std)]))

    batch_size = 100

    test_loader = DataLoader(datasets_masks, batch_size=batch_size, shuffle=False)

    y_pred_test, predictions_df1 = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            log_ps = model(images)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(log_ps)
            y_pred_test += y_pred
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            top_class1 = top_class.view(-1)
            predictions_df1 += top_class1

    prediction_df = pd.DataFrame(predictions_df1)
    prediction_df.to_csv("prediction.csv", index=False, header=True)


if __name__ == '__main__':
    main()

####
