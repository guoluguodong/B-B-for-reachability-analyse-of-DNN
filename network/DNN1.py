from torch import nn



class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 128)  # 输入层：784 -> 128
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # 隐藏层：128 -> 64
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)  # 输出层：64 -> 10（10类）

    def forward(self, x):
        x = x.view(-1, 14 * 14)  # 展平输入
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x