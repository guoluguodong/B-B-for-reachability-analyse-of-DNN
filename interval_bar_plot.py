import os
import re

def extract_float_after_keyword(filepath, keyword="结束"):
    with open(filepath, 'r', encoding='GBK') as file:
        content = file.read()

    # 正则：找到 "结束" 后面的 fmin= 和浮点数
    pattern = re.compile(rf"{re.escape(keyword)}.*?fmin=\s*([-+]?\d*\.\d+|\d+)")
    match = pattern.search(content)

    if match:
        value = float(match.group(1))
        return round(value, 3)  # 保留两位小数
    else:
        return None


# 主程序
base_path = "outputs/DNN2/"
resultsmin = {}
for img_idx in range(1):  # img[0] 到 img[9]
    img_folder = f"img[{img_idx}]/min/"
    full_img_path = os.path.join(base_path, img_folder)

    for label_idx in range(10):  # label0 到 label9
        filename = f"DNN2img{img_idx}_label{label_idx}.log"
        filepath = os.path.join(full_img_path, filename)

        if os.path.exists(filepath):
            value = extract_float_after_keyword(filepath)
            print(img_idx, ' ', label_idx,' ', value)
            resultsmin[(img_idx, label_idx)] = value
        else:
            print(f"文件不存在: {filepath}")
            resultsmin[(img_idx, label_idx)] = None

resultsmax = {}
for img_idx in range(1):  # img[0] 到 img[9]
    img_folder = f"img[{img_idx}]/max/"
    full_img_path = os.path.join(base_path, img_folder)

    for label_idx in range(10):  # label0 到 label9
        filename = f"DNN2img{img_idx}_label{label_idx}.log"
        filepath = os.path.join(full_img_path, filename)

        if os.path.exists(filepath):
            value = extract_float_after_keyword(filepath)
            print(img_idx, ' ', label_idx,' ', -value)
            resultsmax[(img_idx, label_idx)] = -value
        else:
            print(f"文件不存在: {filepath}")
            resultsmax[(img_idx, label_idx)] = None

import matplotlib.pyplot as plt
import numpy as np
for img_idx in range(1):
    # 示例数据
    y_min = []
    y_max = []
    for label_idx in range(10):
        y_min.append(resultsmin[(img_idx, label_idx)])
        y_max.append(resultsmax[(img_idx, label_idx)])
    x = np.arange(10)  # 横坐标0到9
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    # 计算中点和高度
    y_center = (y_min + y_max) / 2
    height = y_max - y_min

    # 画柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, height, bottom=y_min, width=0.5, align='center', color='skyblue', edgecolor='black')

    # 添加最小值和最大值作为标注（可选）
    for i in range(len(x)):
        ax.text(x[i], y_min[i] - 0.2, f'{y_min[i]:.1f}', ha='center', va='top', fontsize=8)
        ax.text(x[i], y_max[i] + 0.2, f'{y_max[i]:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Value')
    ax.set_title('Min-Max Interval Bar Chart')
    # 设置x轴刻度
    ax.set_xticks(np.arange(10))  # 确保每个数字0到9都有显示
    ax.grid(True, linestyle='--', alpha=0.7)
    save_path ='interval_bar_plot\\DNN2\\'+f'interval_bar_plot_{img_idx}.png'
    print(save_path)
    plt.savefig(save_path)
    plt.show()
    plt.close()  # 关闭当前图，防止内存泄漏
