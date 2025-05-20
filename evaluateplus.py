import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from network.DNN2 import DNN2
from network.DNN1 import DNN1
from network.DNN3 import DNN3
from network.DNN4 import DNN4
from network.DNN5 import DNN5
from network.DNN6 import DNN6
from network.DNN7 import DNN7
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cpu")



def extract_info_arrays(log_path):
    """
    从日志中提取 'INFO -[' 开头、以最后一个 ']' 结束的数组字符串，并转换为 numpy 数组列表。

    参数:
        log_path (str): 日志文件路径

    返回:
        List[np.ndarray]: 提取到的数组列表
    """
    # extracted_arrays = []
    numbers_str = ''
    flag = 0
    with open(log_path, 'r') as f:
        for line in f:
            if 'INFO - [' in line:
                # 提取从 'INFO -[' 开始，到最后一个 ']'
                numbers_str += line
                numbers_str = numbers_str[numbers_str.find('['):]
                flag = 1

                continue
            if flag == 1:
                numbers_str += line
            if ']' in line and flag == 1:
                break
    return numbers_str






from torchvision import transforms
for i in range(1,8):
    # 将字符串粘贴到代码中转为 numpy 数组
    if i==2 :
        model = DNN2().to(device)
    elif i==3 :
        model = DNN3().to(device)
    elif i==4 :
        model = DNN4().to(device)
    elif i==5 :
        model = DNN5().to(device)
    elif i==6 :
        model = DNN6().to(device)
    elif i==7 :
        model = DNN7().to(device)
    else:
        model = DNN1().to(device)
    # arr_str = """[ [
    # 1.64537098e-02 3.29685398e-03 5.06289180e-04 5.69149101e-04
    #  1.99946200e-02 2.49080075e-02 7.95042452e-05 3.00918750e-05
    #  2.41000299e-02 4.13289406e-05 4.03067455e-04 2.49869021e-02
    #  3.67366666e-04 1.00530652e-02 2.40333684e-02 2.02790535e-02
    #  6.29954890e-05 9.38498360e-03 2.49805972e-02 1.94290706e-02
    #  4.99580614e-02 4.40962657e-02 3.34309257e-02 2.20850541e-04
    #  1.03346722e-03 4.99958667e-02 6.92080739e-04 1.91942046e-02
    #  4.98341977e-02 4.99999548e-02 4.98912861e-02 4.99708050e-02
    #  1.59664834e-01 3.07006635e-01 2.78210453e-02 4.89475376e-02
    #  2.47797368e-05 1.12059161e-05 4.99136046e-02 4.99829267e-02
    #  3.75478317e-04 7.08247432e-05 9.73924012e-05 2.77974681e-04
    #  4.99998001e-02 4.99513448e-02 4.48327649e-01 8.63876545e-01
    #  2.91321635e-01 4.14756639e-04 1.48760218e-02 4.99975412e-02
    #  4.99855410e-02 4.99929474e-02 4.99829887e-02 4.99825413e-02
    #  1.06286472e-02 9.51488792e-04 1.02586072e-07 2.23754722e-06
    #  3.38088164e-07 2.68953483e-01 5.46122396e-01 7.42191494e-01
    #  3.36273118e-01 4.99122685e-02 2.65128029e-04 4.93925092e-02
    #  4.99413234e-02 4.97352816e-02 4.99939549e-02 4.99938789e-02
    #  4.99999912e-02 4.87563855e-02 4.31703642e-04 1.72069059e-02
    #  4.84381283e-02 2.10160000e-01 9.12744211e-01 2.08879011e-01
    #  2.07666555e-05 2.32229058e-08 7.06687339e-05 2.02669328e-08
    #  4.38024030e-02 4.99825285e-02 4.97059862e-02 4.99601339e-02
    #  4.99940303e-02 4.96168757e-02 8.52915121e-02 3.14746237e-01
    #  7.62451424e-01 2.49833855e-01 4.97404490e-02 4.99804070e-02
    #  4.99515979e-02 4.90262190e-02 5.66331347e-05 9.22919147e-05
    #  2.44109457e-04 4.17996390e-04 4.98675022e-02 3.87229435e-01
    #  7.81372573e-01 5.50397922e-01 2.85275238e-01 4.99643313e-02
    #  4.99316173e-02 4.99325664e-02 4.96229282e-02 4.98072268e-02
    #  4.68937364e-02 4.99779638e-02 5.41476035e-03 4.96834592e-02
    #  9.67368977e-02 9.16665090e-01 8.74528989e-01 7.73545250e-01
    #  5.63724372e-01 7.74172868e-02 1.53163257e-05 1.78727485e-05
    #  2.98139863e-04 2.07370437e-04 4.75635530e-02 4.18948244e-03
    #  2.09367631e-03 4.99999944e-02 4.99789802e-02 8.01666139e-05
    #  5.35825026e-02 1.00842734e-01 8.95130469e-01 1.59625873e-03
    #  3.24643275e-04 8.85697617e-05 6.04621447e-08 5.74223878e-05
    #  1.01255089e-03 2.15205141e-05 4.99026170e-02 4.35672123e-02
    #  1.26078892e-04 1.23658532e-04 4.69304312e-02 4.69530235e-01
    #  4.99222226e-01 3.48552183e-08 4.99946719e-02 4.99760597e-02
    #  4.99292196e-02 4.99742344e-02 1.52911654e-04 4.96244091e-02
    #  4.97007573e-02 7.25413585e-04 8.73224832e-06 1.69621285e-01
    #  4.04925810e-01 7.30397437e-01 2.09276403e-05 2.89905785e-04
    #  9.80729906e-05 4.99696685e-02 4.99933000e-02 4.99949312e-02
    #  1.29472780e-03 4.98989076e-02 2.25931734e-02 5.77840340e-08
    #  4.69192202e-04 8.72586557e-02 4.30688897e-01 8.52885393e-02
    #  4.99804540e-02 3.91520423e-02 1.88397002e-05 1.13094749e-04
    #  2.28049298e-05 1.27272566e-04 4.93592290e-02 1.84450791e-02
    #  4.55072188e-05 2.85685209e-03 4.98920157e-02 4.98605170e-02
    #  4.99995365e-02 4.99581000e-02 2.14452951e-02 4.99421455e-02
    #  4.99760582e-02 4.94070951e-02 4.97728347e-02 4.92742199e-02]
    # ]"""
    label=3
    range = []
    confidence = []
    model.load_state_dict(torch.load(f'network/DNN{i}.pth', map_location='cpu'))
    model.eval()
    image = Image.open(f'img[{label}].png')
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)

    # 设置中文字体，例如 SimHei（黑体）或 Microsoft YaHei（微软雅黑）
    plt.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'

    # 避免负号 '-' 显示为方块
    plt.rcParams['axes.unicode_minus'] = False
    # 读取图片
    arr_str = extract_info_arrays(f'outputsconfidence/DNN{i}/img[{label}]/min/DNN{i}img{label}_label{label}.log')
    # 去掉括号和换行，转换成数组
    arr = np.fromstring(arr_str.replace('[', '').replace(']', '').replace('\n', ' '), sep=' ')

    # 确保长度合适（如14x14或16x16）
    if arr.size == 196:
        imgmin = arr.reshape(14, 14)
    else:
        raise ValueError(f"Array size {arr.size} not supported for reshaping")
    arr_str = extract_info_arrays(f'outputsconfidence/DNN{i}/img[{label}]/max/DNN{i}img{label}_label{label}.log')
    # 去掉括号和换行，转换成数组
    arr = np.fromstring(arr_str.replace('[', '').replace(']', '').replace('\n', ' '), sep=' ')

    # 确保长度合适（如14x14或16x16）
    if arr.size == 196:
        imgmax = arr.reshape(14, 14)
    else:
        raise ValueError(f"Array size {arr.size} not supported for reshaping")


    # f_origin
    out = model(image_tensor.float())[0]
    print(out)
    prob = F.softmax(out)
    print(prob)
    range.append(float("{:.4g}".format(out[label].item())))
    confidence.append(float("{:.4g}".format(prob[label].item())))
    # f_lb
    out = model(torch.as_tensor(imgmin).float())[0]
    print(out)
    prob = F.softmax(out)
    print(prob)
    range.append(float("{:.4g}".format(out[label].item())))
    confidence.append(float("{:.4g}".format(prob[label].item())))
    # f_ub
    out = model(torch.as_tensor(imgmax).float())[0]
    print(out)
    prob = F.softmax(out)
    print(prob)

    range.append(float("{:.4g}".format(out[label].item())))
    confidence.append(float("{:.4g}".format(prob[label].item())))
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    plt.subplots_adjust(wspace=0)  # wspace 控制横向子图之间的间距

    # 分别在每个子图上绘制图像
    axs[0].imshow(image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title(f"原始图片\n{range[0]}\n{confidence[0]}")

    axs[1].imshow(image, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f"最小输出图片\n{range[1]}\n{confidence[1]}")

    axs[2].imshow(image, cmap='gray')
    axs[2].axis('off')
    axs[2].set_title(f"最大输出图片\n{range[2]}\n{confidence[2]}")

    # plt.tight_layout()
    plt.savefig(f"result/dnn{i}image{[label]}_lbub.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # device = torch.device("cpu")
    # model = DNN7().to(device)
    # model.load_state_dict(torch.load('network/DNN7.pth', map_location='cpu'))
    # model.eval()
    # out = model(torch.as_tensor(img).float())[0]
    # print(out)
    # prob = F.softmax(out)
    # print(prob)
    # # 画图并保存
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    # plt.show()


