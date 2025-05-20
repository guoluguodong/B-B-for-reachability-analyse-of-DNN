# 3. 定义简单的三层神经网络
import torch
from torch import nn


class DNN2(nn.Module):
    def __init__(self):
        super(DNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 14, 14)  # 恢复成 Conv2d 能接受的形状
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 3600)
        x = self.fc1(x)
        x = self.relu(x)
        x=self.fc2(x)
        return x
