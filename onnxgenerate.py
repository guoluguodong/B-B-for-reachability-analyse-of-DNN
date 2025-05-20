import torch
from network.DNN1 import   DNN1
import scipy.io as sio
device = torch.device("cpu")
model=DNN1().to(device)
# 假设你已经加载了训练好的模型
model.eval()
# 假设你已经定义并加载好了模型
model.eval()

# 提取参数并转为 numpy
params = {}
for name, param in model.named_parameters():
    params[name] = param.detach().cpu().numpy()

# 保存为 .mat 文件
sio.savemat('model_weights.mat', params)
