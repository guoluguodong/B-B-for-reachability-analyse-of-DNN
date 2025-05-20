# 3. 定义简单的三层神经网络
import torch
from torch import nn


class DNN7(nn.Module):
    def __init__(self):
        super(DNN7, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, padding=1)
        # self.norm1 = nn.BatchNorm1d(16)
        self.relu2= nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, padding=1)
        self.relu3= nn.ReLU()
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        # self.norm2 = nn.BatchNorm1d(64)
        self.relu4= nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(20736, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 14, 14)  # 恢复成 Conv2d 能接受的形状
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # x = self.norm1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        # x = self.norm2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # x = self.norm3(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = x.view(-1, 20736)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
