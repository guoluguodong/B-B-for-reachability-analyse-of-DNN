import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN3(nn.Module):
    def __init__(self):
        super(DNN3, self).__init__()
        # 输入: (1, 14, 14)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 输出: (16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # 输出: (16, 7, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 输出: (32, 7, 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设10类分类任务

    def forward(self, x):
        x = x.view(-1, 1, 14, 14)  # 恢复成 Conv2d 能接受的形状
        x = self.pool(F.relu(self.conv1(x)))  # (16, 7, 7)
        x = F.relu(self.conv2(x))             # (32, 7, 7)
        x = x.view(x.size(0), -1)
        # 展平为 (batch_size, 32*7*7)
        x = x.view(-1, 1568)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 shape: (batch_size, 10)
        return x
