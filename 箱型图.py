
from PIL import Image
from docutils.writers.odf_odt import ToString
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from network.DNN1 import DNN1
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_epsilon_bound(x, epsilon=0.05):
    """给定输入x，返回其epsilon邻域的上下界"""
    lower_bound = torch.clamp(x - epsilon, 0.0, 1.0)
    upper_bound = torch.clamp(x + epsilon, 0.0, 1.0)
    return lower_bound, upper_bound

device = torch.device("cpu")
model = DNN1().to(device)
model.load_state_dict(torch.load('network/DNN1.pth', map_location='cpu'))
model.eval()
torch.manual_seed(42)
for label in range(0,10):
    image = Image.open('img['+str(label)+'].png')
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)
    x_bound = get_epsilon_bound(image_tensor.view(-1, 14 * 14), epsilon=0.05)
    outputs_for_each_label = [[] for _ in range(10)]  # 10个类别：0~9
    for i in range(1000):
        random_sample = torch.empty_like(x_bound[0]).uniform_(0, 1) * (x_bound[1] - x_bound[0]) + x_bound[0]
        # 输入模型，注意reshape回原shape
        output = model(random_sample.view(-1, 1, 14, 14))
        print(output)
        for i in range(10):
            score = output[0, i].item()
            outputs_for_each_label[i].append(score)
    # 绘制箱型图
    plt.figure(figsize=(10, 6))
    plt.boxplot(outputs_for_each_label, vert=True, labels=[str(i) for i in range(10)])
    plt.xlabel('Label')
    plt.ylabel('Model Output Score')
    plt.title('Boxplot of Model Outputs for Labels 0 to 9')
    plt.grid(True)
    # 保存图像
    save_path ='箱型图\\DNN1\\'+f'boxplot_label_{label}.png'
    print(f"Saved boxplot for label {label} to {save_path}")
    plt.savefig(save_path)
    plt.show()
    plt.close()  # 关闭当前图，防止内存泄漏