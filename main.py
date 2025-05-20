""""
核心算法
# def iterate(root: TreeNode) -> bool:
#     queue = []
#
#     queue.append(root)
#
#     while queue:
#         tree_node = queue.pop(0)
#
#         while tree_node.parent is not None and tree_node.k > tree_node.parent.k:
#             queue = [tree_node.parent]
#
#         if tree_node.left is None:
#         # get_left 和 get_right 计算下一层的 k
#             tree_node.left = TreeNode(k=None, left=None, right=None, parent=tree_node, interval=get_sub_inteval(tree_node)["left"])
#         if tree_node.right is None:
#             tree_node.right = TreeNode(k=None, left=None, right=None, parent=tree_node, interval=get_sub_inteval(tree_node)["right"])
#
#         k_left = get_k(left)
#         k_right = get_k(right)
#
#         # 回退
#         if k_left > tree_node.k or k_right > tree_node.k:
#             tree_node.k = max(k_left, k_right)
#             queue = [tree_node.parent]
#         else:
#             tree_node.left.k = k_left
#             tree_node.right.k = k_right
#             queue.append(tree_node.left)
#             queue.append(tree_node.right)
queue为堆
"""
import argparse
import heapq
import logging
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from scipy.optimize import linprog

from network.DNN2 import DNN2
from network.DNN1 import DNN1
from utils.AdaLIPO_P import AdaLIPO_P
from utils.BoxTreeNode import BoxTreeNode, PrioritizedItem
import os
from utils.logConfig import load_logging_config
from utils.util import generate_vertex_and_opposite, is_ancestor, print_all_nodes, get_epsilon_neighborhood

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cma
import numpy as np


def bisect(x):
    # 选择具有最大宽度的维度，并通过中间点对其进行二分。 也可以采用启发式算法
    start, end = x[0], x[1]
    # 计算每个维度的宽度
    mid = (start + end) / 2
    grad = estimate_gradient(mid)
    n = 20
    max_dim = np.argsort(grad)[:n]  # 排序并反向获取最大的n个索引
    # 生成两个新区间
    mid1 = start.copy()
    mid2 = end.copy()
    mid1[max_dim] = mid[max_dim]
    mid2[max_dim] = mid[max_dim]
    x1 = np.array([start.tolist(), mid2.tolist()])
    x2 = np.array([mid1.tolist(), end.tolist()])
    return x1, x2, max_dim

import torch.nn.functional as F
def black_function(x):
    if min:
        out = model(torch.as_tensor(x).float()).detach().numpy()[0]
        probs = F.softmax(out, dim=0)  # 计算概率
        return probs[label].item()
    else:
        out = -model(torch.as_tensor(x).float()).detach().numpy()[0]
        return out[label]

def estimate_gradient(x0, epsilon=1e-6):
    n = len(x0)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_plus = black_function(x0 + epsilon * e_i)
        f_minus = black_function(x0 - epsilon * e_i)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad


def LinearRelaxation(x, k, dim):
    A_ub, b_ub = [], []
    x_min, x_max = x[0], x[1]
    for i in range(100):
        A_ub.append([-k, -1])
        A_ub.append([k, -1])
        vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
        x0 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
        x1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(opposite_vertex)])
        x0[dim] = x_min[dim]
        x1[dim] = x_max[dim]
        sum_x0, sum_x1 = sum(x0), sum(x1)
        # fx0 = model(torch.as_tensor(x0).cuda().float()).cpu().detach().numpy()[0][0]
        # fx1 = model(torch.as_tensor(x1).cuda().float()).cpu().detach().numpy()[0][0]
        fx0 = black_function(x0)
        fx1 = black_function(x1)
        # b_ub.append(-k * sum_x0 - fx0)
        # b_ub.append(k * sum_x1 - fx1)
        if sum_x0 < sum_x1:
            b_ub.append(-k * sum_x0 - fx0)
            b_ub.append(k * sum_x1 - fx1)
        else:
            b_ub.append(-k * sum_x1 - fx1)
            b_ub.append(k * sum_x0 - fx0)
    return {
        "A_ub": A_ub,
        "b_ub": b_ub
    }


def PolytopeHull(x, k, dim):
    # 计算目标函数的线性松弛
    fl = LinearRelaxation(x, k, dim)
    A_ub = fl['A_ub']
    b_ub = fl['b_ub']
    # 计算下界
    result = linprog([0, 1], A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(sum(x[0]), sum(x[1])), (None, None)], method='highs')
    lbx = result.fun
    return {'x': x, 'lbx': lbx, 'success': result.success}


def contract_and_bound(node: BoxTreeNode, epsilon, k_hat, dim):
    x_lbx = PolytopeHull(node.interval, k_hat, dim)
    x, lbx = x_lbx['x'], x_lbx['lbx']
    if not x_lbx['success']:
        raise ValueError("线性规划求解失败")
    return {
        'x': x,
        'lbx': lbx,
    }


def interval_branch_and_bound(x_initial, epsilon, ep=1):
    # 初始化root
    global left_x, right_x, dim
    root = BoxTreeNode(k=0, left=None, right=None, parent=None, interval=x_initial, f_min=-np.inf, f_best=np.inf)
    # root
    result = run_cma_es(objective_function=black_function, bounds_array=x_initial, sigma0=0.001, popsize=40,
                        maxiter=200)
    root_f_best, root_x = result['best_f'], result['best_x']
    # root_f_best, root_x = FeasibleSearch( root.interval, 50)
    k_hat, _ = AdaLIPO_P(model, root.interval, x=root_x, n=100, black_function=black_function)
    result_root = contract_and_bound(root, epsilon, k_hat, 0)
    while root_f_best < result_root['lbx']:
        k_hat = k_hat * 1.2
        # logging.info(f"重算右子节点k,k={result_right['k_hat']},result_right['f_best'{result_right['f_best']} , result_right['lbx']{result_right['lbx']}")
        result_root = contract_and_bound(root, epsilon, k_hat, 0)
    root.k, root.f_min = k_hat, result_root['lbx']
    root.f_best = root_f_best
    queue = []
    heapq.heappush(queue, PrioritizedItem(root.f_best, root))
    node_number = 1
    f_best_current = np.inf
    f_min = -np.inf
    while queue:
        tree_node = heapq.heappop(queue).item
        logging.info(f"cur_node_number {node_number},f_best {tree_node.f_best},f_min {tree_node.f_min},k {tree_node.k}")
        # 更新 f_best_current f_min
        if tree_node.f_best <= f_best_current:
            f_best_current = tree_node.f_best
            f_min = tree_node.f_min
            if abs(tree_node.f_best - tree_node.f_min) / abs(tree_node.f_min) < epsilon:
                logging.info(f"f_best_current {f_best_current},f_min {f_min}")
                break
        logging.info(f"f_best_current {f_best_current},f_min {f_min}")

        while tree_node.parent is not None and tree_node.k > tree_node.parent.k:
            tree_node.parent.k = tree_node.k
            tree_node = tree_node.parent
            f_min = tree_node.f_min
            f_best_current = tree_node.f_best
            queue = [item for item in queue if not is_ancestor(item.item, tree_node)]
            heapq.heapify(queue)
            logging.info(f"回退,len(queue)={len(queue)}")
        x1, x2, dim = bisect(root.interval)
        tree_node.left = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
                                     interval=x1)
        tree_node.right = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
                                      interval=x2)

        # if tree_node.left is None or tree_node.right is None:
        #
        #     x1, x2,dim = bisect(root.interval)
        #     tree_node.left = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
        #                                  interval=x1)
        #     tree_node.right = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
        #                                   interval=x2)
        right_f_best = np.inf
        left_f_best = np.inf
        # 随机搜索 最小值,至少有一个必须小于等于父节点的最小值，才允许终止搜索
        i = 0
        while right_f_best > tree_node.f_best * ep and left_f_best > tree_node.f_best * ep:
            result1 = run_cma_es(objective_function=black_function, bounds_array=tree_node.left.interval, sigma0=0.001,
                                 popsize=40,
                                 maxiter=200 + i * 10)
            left_f_best, left_x = result1['best_f'], result1['best_x']
            result2 = run_cma_es(objective_function=black_function, bounds_array=tree_node.right.interval, sigma0=0.001,
                                 popsize=40,
                                 maxiter=200 + i * 10)
            right_f_best, right_x = result2['best_f'], result2['best_x']
            if i > 10:
                if left_f_best > right_f_best:
                    right_f_best = tree_node.f_best * ep
                else:
                    left_f_best = tree_node.f_best * ep
            # left_f_best,left_x = FeasibleSearch(model,tree_node.left.interval,50)
            # right_f_best,right_x = FeasibleSearch(model, tree_node.right.interval, 50)
            i += 1
            logging.info(
                f"tree_node.f_best*ep:{tree_node.f_best * ep},left_f_best{left_f_best},right_f_best{right_f_best},i:{i}")
            # if i%10 ==0:
            #     logging.info(f"重新{i}次搜索最小值")
        tree_node.left.f_best = left_f_best
        tree_node.right.f_best = right_f_best
        # left
        k_hat, _ = AdaLIPO_P(model, tree_node.left.interval, left_x, n=100, black_function=black_function)
        result_left = contract_and_bound(tree_node.left, epsilon, k_hat, dim)
        while left_f_best < result_left['lbx']:
            k_hat = k_hat * 1.01
            # logging.info(f"重算左子节点k,k={result_left['k_hat']},result_left['f_best'{result_left['f_best']} , result_left['lbx']{result_left['lbx']}")
            result_left = contract_and_bound(tree_node.left, epsilon, k_hat, dim)
        tree_node.left.k, tree_node.left.f_min = k_hat, result_left['lbx']
        # right
        k_hat, _ = AdaLIPO_P(model, tree_node.right.interval, right_x, n=100, black_function=black_function)
        result_right = contract_and_bound(tree_node.right, epsilon, k_hat, dim)
        while right_f_best < result_right['lbx']:
            k_hat = k_hat * 1.01
            # logging.info(f"重算右子节点k,k={result_right['k_hat']},result_right['f_best'{result_right['f_best']} , result_right['lbx']{result_right['lbx']}")
            result_right = contract_and_bound(tree_node.right, epsilon, k_hat, dim)
        tree_node.right.k, tree_node.right.f_min = k_hat, result_right['lbx']

        heapq.heappush(queue, PrioritizedItem(tree_node.left.f_best, tree_node.left))
        heapq.heappush(queue, PrioritizedItem(tree_node.right.f_best, tree_node.right))

        logging.info(f"添加左子节点，右子节点，len(queue):{len(queue)}")
        print_all_nodes(root, logging)
        node_number += 1
    return f_min, f_best_current


# ===================================================================
# 启发式搜索
def convert_bounds(bounds_array):
    """
    将一个 2 × n 的 numpy.array 转为 CMA-ES 所需的 [lower_bounds, upper_bounds] 格式
    """
    assert bounds_array.shape[0] == 2, "bounds_array 应为 2 × n 的形状"
    lower_bounds = bounds_array[0, :].tolist()
    upper_bounds = bounds_array[1, :].tolist()
    return [lower_bounds, upper_bounds]


def run_cma_es(
        objective_function,
        bounds_array,  # List of (lower, upper) tuples, one per dimension
        x0=None,  # Optional initial solution
        sigma0=0.5,  # Initial step size
        popsize=40,  # Population size
        maxiter=200  # Max number of iterations
):
    # 转换为 CMA-ES 格式
    bounds_cma = convert_bounds(bounds_array)
    # 使用 x0 时要确保落在范围内
    x0 = np.mean(bounds_array, axis=0)  # 每维中点
    # 调用优化
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': bounds_cma,
        'popsize': popsize,
        'maxiter': maxiter,
        'verb_disp': 0,
    })
    es.optimize(objective_function)

    # 返回结果
    return {
        'best_x': es.result.xbest,
        'best_f': es.result.fbest,
        'evolution_path': es.logger
    }


# ===================================================================

# def main(args):
#     load_logging_config(args.log_file)
#     logging.info(args)
#     device = torch.device("cpu")
#     global model,label,min
#     model = DNN1().to(device)
#     label = args.label
#     model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
#     model.eval()
#     min = True if args.min==1 else False
#     image = Image.open(args.image_path)
#     transform = transforms.ToTensor()
#     image_tensor = transform(image).to(device)
#     outputs = model(image_tensor)
#     logging.info(f'Model output max(1): {outputs.max(1)}')
#     x_initial = get_epsilon_neighborhood(image_tensor.view(-1, 14 * 14), epsilon=args.init_epsilon)
#     # 调用优化算法
#     fmin, fbestcurrent = interval_branch_and_bound(x_initial, args.epsilon,ep =args.ep)
#     logging.info(f"结束,fmin= {fmin}, fbestmin={fbestcurrent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model prediction and interval optimization.")
    parser.add_argument("--log_file", type=str, default="outputs/my_custom_log.log", help="Path to the log file")
    parser.add_argument("--image_path", type=str, default='img[2].png',
                        help="Path to the input image (e.g., 'img[2].png')")
    parser.add_argument("--model_path", type=str, default="network/DNN1.pth", help="Path to the model checkpoint")
    parser.add_argument("--label", type=int, default=0, help="Label to use for optimization (if needed)")
    parser.add_argument("--init_epsilon", type=float, default=0.05, help="Initial neighborhood epsilon")
    parser.add_argument("--epsilon", type=float, default=5e-3, help="Precision requirement for optimization")
    parser.add_argument("--ep", type=float, default=1, help="Precision requirement for optimization")
    parser.add_argument("--min", type=int, default=1, help="求解最大值还是最小值")
    args = parser.parse_args()
    # main(args)
    for i in range(0,10):
        device = torch.device("cpu")
        model = DNN1().to(device)
        range = []
        confidence = []
        label = i
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.eval()
        image = Image.open(f'img[{i}].png')
        transform = transforms.ToTensor()
        image_tensor = transform(image).to(device)
        outputs = model(image_tensor)[0]
        probs = F.softmax(outputs, dim=0)
        range.append(float("{:.4g}".format(outputs[label].item())))
        confidence.append(float("{:.4g}".format(probs[label].item())))
        # 设置中文字体，例如 SimHei（黑体）或 Microsoft YaHei（微软雅黑）
        plt.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'

        # 避免负号 '-' 显示为方块
        plt.rcParams['axes.unicode_minus'] = False
        

        image = image_tensor.squeeze()  # => (14, 14)
        x_initial = get_epsilon_neighborhood(image_tensor.view(-1, 14 * 14), epsilon=args.init_epsilon)
        x_lb, x_ub = x_initial[0], x_initial[1]
        x_lb = x_lb.reshape(14, 14)
        x_ub = x_ub.reshape(14, 14)
        outputs1 = model(torch.as_tensor(x_lb).float()).detach().numpy()[0]
        outputs1 = torch.tensor(outputs1)
        probs = F.softmax(outputs1, dim=0)
        range.append(float("{:.4g}".format(outputs1[label].item())))
        confidence.append(float("{:.4g}".format(probs[label].item())))

        outputs2 = model(torch.as_tensor(x_ub).float()).detach().numpy()[0]
        outputs2 = torch.tensor(outputs2)
        probs = F.softmax(outputs2, dim=0)
        range.append(float("{:.4g}".format(outputs2[label].item())))
        confidence.append(float("{:.4g}".format(probs[label].item())))

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        plt.subplots_adjust(wspace=0)  # wspace 控制横向子图之间的间距

        # 分别在每个子图上绘制图像
        axs[0].imshow(image.squeeze(), cmap='gray')
        axs[0].axis('off')
        axs[0].set_title(f"原始图片\n{range[0]}\n{confidence[0]}")

        axs[1].imshow(x_lb.squeeze(), cmap='gray')
        axs[1].axis('off')
        axs[1].set_title(f"扰动下界图片\n{range[1]}\n{confidence[1]}")

        axs[2].imshow(x_ub.squeeze(), cmap='gray')
        axs[2].axis('off')
        axs[2].set_title(f"扰动上界图片\n{range[2]}\n{confidence[2]}")

        # plt.tight_layout()
        plt.savefig(f"image{i}_lbub.png", bbox_inches='tight', pad_inches=0)
        plt.close()


