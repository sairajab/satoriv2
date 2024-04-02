import math
import torch
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv_kernel_size1 = 13
        conv_kernel_size2 = 8
        pool_kernel_size1 = 4
        pool_kernel_size2 = 3
        self.motif_scanning_layer = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=256, kernel_size=conv_kernel_size1, padding=0, bias=False),
                                                  nn.ReLU(inplace=True),
                                                  nn.MaxPool1d(kernel_size=pool_kernel_size1))
        self.conv_net = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1,
                         stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2,
                         stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        self.num_channels = 512

    def forward(self, x):
        out = self.motif_scanning_layer(x)
        out = self.conv_net(out)
        return out


class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        conv_kernel_size1 = 13
        conv_kernel_size2 = 8
        pool_kernel_size1 = 4
        pool_kernel_size2 = 4
        self.motif_scanning_layer = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=256, kernel_size=conv_kernel_size1, padding=0, bias=False),
                                                  nn.BatchNorm1d(
                                                      num_features=256),
                                                  nn.ReLU(),
                                                  nn.MaxPool1d(kernel_size=pool_kernel_size1))
        self.conv_net = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size2,  padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2,
                         stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2,  padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        self.num_channels = 512

    def forward(self, x):

        out = self.motif_scanning_layer(x)
        out = self.conv_net(out)
        return out


class PretrainCNN(nn.Module):

    def __init__(self, no_layers, num_labels=2):
        super(PretrainCNN, self).__init__()
        if no_layers == 4:
            self.cnn_module = CNN4()
        else:
            self.cnn_module = CNN()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(36*512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.4),
                                 nn.Linear(256, num_labels))  # [12 for arabidopsis using big network][using 4 layers=- 36 for arabidpsis, 18 for simulated

    def forward(self, x):
        output = self.cnn_module(x)
        output = self.flatten(output)
        output = self.mlp(output)

        return output
