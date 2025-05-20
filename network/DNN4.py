from torch import nn
import torch.nn.functional as F


class DNN4(nn.Module):
    def __init__(self):
        super(DNN4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (32, 14, 14)
        self.pool = nn.MaxPool2d(2, 2)                           # (32, 7, 7)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 14, 14)  # 恢复成 Conv2d 能接受的形状
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 1568)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x