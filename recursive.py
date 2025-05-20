import heapq
import random
from collections import deque
from idlelib.tree import TreeNode
from random import sample

import numpy as np
import torch
from scipy.optimize import linprog

from network.DNN1 import SimpleNN14
from utils import *
import dataclasses
from typing import List, Optional, Any
import os

from utils.LIPOUtil import Bernoulli, Uniform

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from torch import nn

model = SimpleNN14().to(device)
model.load_state_dict(torch.load("network/DNN1.pth"))
model.eval()  # 进入推理模式
state_dict = model.state_dict()

from PIL import Image
from torchvision import transforms

# 读取图像文件
image = Image.open('img[0].png')
# 定义转换操作
transform = transforms.ToTensor()
# 将PIL图像转换为Tensor
image_tensor = transform(image).to(device)
# 打印Tensor信息
print(type(image_tensor))  # <class 'torch.Tensor'>
print(image_tensor.shape)  # 打印Tensor的形状
print(image_tensor.min(), image_tensor.max())  # 打印Tensor的最小值和最大值
outputs = model(image_tensor)
print(outputs.max(1))


def get_epsilon_neighborhood(tensor: torch.Tensor, epsilon: float):
    tensor = tensor.squeeze(0)  # 去掉第一个维度，变成 (28, 28)
    neighborhood = np.array([
        [min(max((val - epsilon).item(), 0.0), 1.0) for val in tensor],
        [min(max((val + epsilon).item(), 0.0), 1.0) for val in tensor]
    ])
    return neighborhood


def best_box(L):
    lb_min = np.inf
    selected_order = 0
    for (i, d) in enumerate(L):
        if d['lbx'] < lb_min:
            selected_order = i
            lb_min = d['lbx']
    return selected_order


def generate_vertex_and_opposite(n):
    # 生成长度为 n 的随机 01 序列
    vertex = [random.randint(0, 1) for _ in range(n)]
    # 生成相对的顶点：0 变 1，1 变 0
    opposite_vertex = [1 - bit for bit in vertex]
    return vertex, opposite_vertex


def bisect(x):
    # 选择具有最大宽度的维度，并通过中间点对其进行二分。 也可以采用启发式算法
    start, end = x[0], x[1]
    # 计算每个维度的宽度
    widths = end - start
    # 找到最大宽度的维度索引
    max_dim = np.argmax(widths)
    # 计算该维度的中点
    mid = (start[max_dim] + end[max_dim]) / 2
    # 生成两个新区间
    mid1 = start.copy()
    mid2 = end.copy()
    mid1[max_dim] = mid
    mid2[max_dim] = mid
    x1 = np.array([start.tolist(), mid2.tolist()])
    x2 = np.array([mid1.tolist(), end.tolist()])
    return x1, x2
def precompute_vertices_and_values(x_min, x_max, model, num_pairs=100):
    samples = []
    for _ in range(num_pairs):
        vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
        x0 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
        x1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(opposite_vertex)])
        fx0 = model(torch.as_tensor(x0).float()).detach().numpy()[0][0]
        fx1 = model(torch.as_tensor(x1).float()).detach().numpy()[0][0]
        samples.append((x0, fx0, x1, fx1))
    return samples

def LinearRelaxation(k, samples):
    A_ub, b_ub = [], []
    for x0, fx0, x1, fx1 in samples:
        A_ub.append([-k, -1])
        A_ub.append([k, -1])
        sum_x0, sum_x1 = sum(x0), sum(x1)
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

#
# def LinearRelaxation(x, model, k):
#     A_ub, b_ub = [], []
#     x_min, x_max = x[0], x[1]
#     for i in range(1000):
#         A_ub.append([-k, -1])
#         A_ub.append([k, -1])
#         vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
#         x0 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
#         x1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(opposite_vertex)])
#         sum_x0, sum_x1 = sum(x0), sum(x1)
#         # fx0 = model(torch.as_tensor(x0).cuda().float()).cpu().detach().numpy()[0][0]
#         # fx1 = model(torch.as_tensor(x1).cuda().float()).cpu().detach().numpy()[0][0]
#         fx0 = model(torch.as_tensor(x0).float()).detach().numpy()[0][0]
#         fx1 = model(torch.as_tensor(x1).float()).detach().numpy()[0][0]
#         # b_ub.append(-k * sum_x0 - fx0)
#         # b_ub.append(k * sum_x1 - fx1)
#         if sum_x0 < sum_x1:
#             b_ub.append(-k * sum_x0 - fx0)
#             b_ub.append(k * sum_x1 - fx1)
#         else:
#             b_ub.append(-k * sum_x1 - fx1)
#             b_ub.append(k * sum_x0 - fx0)
#     return {
#         "A_ub": A_ub,
#         "b_ub": b_ub
#     }


def PolytopeHull(x, model, k,sample):
    # 计算目标函数的线性松弛
    fl = LinearRelaxation(k,sample)
    A_ub = fl['A_ub']
    b_ub = fl['b_ub']
    # 计算下界
    result = linprog([0, 1], A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=np.array([(sum(x[0]), sum(x[1])), (None, None)]), method='highs')
    lbx = result.fun
    return {'x': x, 'lbx': lbx, 'success': result.success}


@dataclasses.dataclass
class BoxTreeNode:
    k: Optional[float] = None
    left: Optional["BoxTreeNode"] = None
    right: Optional["BoxTreeNode"] = None
    parent: Optional["BoxTreeNode"] = None
    interval: Optional[np.ndarray] = None
    f_min: Optional[float] = None
    f_best: Optional[float] = None
@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = dataclasses.field(compare=False)

def contract_and_bound(node: BoxTreeNode, model, epsilon,k_hat,sample):
    x_lbx = PolytopeHull(node.interval, model, k_hat,sample)
    x, lbx = x_lbx['x'], x_lbx['lbx']
    if not x_lbx['success']:
        raise ValueError("线性规划求解失败")

    return {
        'x': x,
        'lbx': lbx,
    }


import dataclasses
from typing import List, Optional


#
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


def update_boxes_lb(L, result1, result2, f_best_current, f_min):
    L.append(
        {'x': result1['x'], 'lbx': result1['lbx'], 'k_hat': result1['k_hat'], 'f_best_in': result1['f_best_current']})
    L.append(
        {'x': result2['x'], 'lbx': result2['lbx'], 'k_hat': result2['k_hat'], 'f_best_in': result2['f_best_current']})
    # 这里我们只保留那些下界大于目标值 f_min 的区间
    new_L = []
    for d in L:
        if (d['lbx'] < f_best_current):
            new_L.append(d)
            f_min = max(d['lbx'], f_min)
    return new_L, f_min


def is_ancestor(node, maybeAncestorNode) -> bool:
    while node.parent is not None:
        node = node.parent
        if node is maybeAncestorNode:
            return True
    return False

def print_all_nodes(root):
    queue = deque([root])
    node_number = 1
    while queue:
        node = queue.popleft()
        print(f"Node {node_number}: k={node.k}, f_best={node.f_best}, f_min={node.f_min}")
        node_number += 1
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

def interval_branch_and_bound(model, x_initial, epsilon):
    # 初始化线性松弛
    sample = precompute_vertices_and_values(x_initial[0], x_initial[1], model)
    # 初始化root
    global left_x, right_x
    root = BoxTreeNode(k=0, left=None, right=None, parent=None, interval=x_initial, f_min=-np.inf, f_best=np.inf)
    # root
    result =run_cma_es(objective_function=rastrigin,bounds_array=x_initial,sigma0=0.001, popsize=40,maxiter=200)
    root_f_best, root_x = result['best_f'],result['best_x']
    # root_f_best, root_x = FeasibleSearch(model, root.interval, 50)
    k_hat, _ = AdaLIPO_P(model, root.interval, x = root_x, n=100)
    result_root = contract_and_bound(root, model, epsilon, k_hat,sample)
    while root_f_best < result_root['lbx']:
        k_hat = k_hat * 1.2
        # print(f"重算右子节点k,k={result_right['k_hat']},result_right['f_best'{result_right['f_best']} , result_right['lbx']{result_right['lbx']}")
        result_root = contract_and_bound(root, model, epsilon, k_hat,sample)
    root.k, root.f_min = k_hat, result_root['lbx']
    root.f_best =root_f_best
    queue = []
    heapq.heappush(queue, PrioritizedItem(root.f_best, root))
    node_number = 1
    f_best_current = np.inf
    f_min = -np.inf

    while queue:
        tree_node = heapq.heappop(queue).item
        print(f"cur_node_number {node_number},f_best {tree_node.f_best},f_min {tree_node.f_min},k {tree_node.k}")
        # 更新 f_best_current f_min
        if tree_node.f_best <= f_best_current:
            f_best_current = tree_node.f_best
            f_min = tree_node.f_min
            if abs(tree_node.f_best - tree_node.f_min) < epsilon:
                print(f"f_best_current {f_best_current},f_min {f_min}")
                break
        print(f"f_best_current {f_best_current},f_min {f_min}")

        while tree_node.parent is not None and tree_node.k > tree_node.parent.k:
            tree_node.parent.k = tree_node.k
            tree_node = tree_node.parent
            queue = [item for item in queue if not is_ancestor(item.item, tree_node)]
            heapq.heapify(queue)

            print(f"回退,len(queue)={len(queue)}")
        if tree_node.left is None or tree_node.right is None:
            x1, x2 = bisect(root.interval)
            tree_node.left = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
                                         interval=x1)
            tree_node.right = BoxTreeNode(k=None, left=None, right=None, parent=tree_node,
                                          interval=x2)
        right_f_best = np.inf
        left_f_best = np.inf
        # 随机搜索 最小值,至少有一个必须小于等于父节点的最小值，才允许终止搜索
        i=0
        while right_f_best >= tree_node.f_best*1.05 or left_f_best >= tree_node.f_best*1.05:
            result1 = run_cma_es(objective_function=rastrigin, bounds_array=tree_node.left.interval, sigma0=0.001, popsize=40,
                                maxiter=200)
            left_f_best, left_x = result1['best_f'], result1['best_x']
            result2 = run_cma_es(objective_function=rastrigin, bounds_array=tree_node.right.interval, sigma0=0.001, popsize=40,
                                maxiter=200)
            right_f_best, right_x = result2['best_f'], result2['best_x']
            # left_f_best,left_x = FeasibleSearch(model,tree_node.left.interval,50)
            # right_f_best,right_x = FeasibleSearch(model, tree_node.right.interval, 50)
            i += 1
            print(f"重新{i}次搜索最小值")
        tree_node.left.f_best = left_f_best
        tree_node.right.f_best = right_f_best
        # left
        k_hat, _ = AdaLIPO_P(model, tree_node.left.interval, left_x,n= 100)
        result_left =  contract_and_bound(tree_node.left, model, epsilon,k_hat,sample)
        while left_f_best < result_left['lbx']:
            k_hat = k_hat*1.01
            # print(f"重算左子节点k,k={result_left['k_hat']},result_left['f_best'{result_left['f_best']} , result_left['lbx']{result_left['lbx']}")
            result_left = contract_and_bound(tree_node.left, model, epsilon,k_hat,sample)
        tree_node.left.k, tree_node.left.f_min = k_hat,result_left['lbx']
        # right
        k_hat, _ = AdaLIPO_P(model, tree_node.right.interval, right_x, n=100)
        result_right = contract_and_bound(tree_node.right, model, epsilon, k_hat,sample)
        while right_f_best < result_right['lbx']:
            k_hat = k_hat*1.01
            # print(f"重算右子节点k,k={result_right['k_hat']},result_right['f_best'{result_right['f_best']} , result_right['lbx']{result_right['lbx']}")
            result_right = contract_and_bound(tree_node.right, model, epsilon,k_hat,sample)
        tree_node.right.k, tree_node.right.f_min = k_hat, result_right['lbx']


        heapq.heappush(queue, PrioritizedItem(tree_node.left.f_best, tree_node.left))
        heapq.heappush(queue, PrioritizedItem(tree_node.right.f_best, tree_node.right))

        print(f"添加左子节点，右子节点，len(queue):{len(queue)}")
        print_all_nodes(root)
        node_number += 1
    return f_min, f_best_current


# def recursive_optimize(L, f_best_current, f_min, model, epsilon, epoch=0):
#     if f_best_current - f_min <= epsilon or not L:
#         print("终止：f_best - f_min =", f_best_current - f_min)
#         return f_best_current
#
#     # 选择当前最优的区间
#     selected_order = best_box(L)
#     x_selected = L[selected_order]['x']
#     k_selected = L[selected_order]['k_hat']
#     f_min = L[selected_order]['lbx']
#     f_best_in = L[selected_order]['f_best_in']
#
#     # 区间二分
#     x1, x2 = bisect(x_selected)
#
#     # 子区间计算
#     result1 = contract_and_bound(x1, model, f_best_current, epsilon, k_selected, f_best_in, f_min)
#     result2 = contract_and_bound(x2, model, f_best_current, epsilon, k_selected, f_best_in, f_min)
#
#     if result1['k_hat'] > k_selected:
#         print("k_hat", result1['k_hat'], "k_selected ", k_selected)
#         back1 = True
#     if result2['k_hat'] > k_selected:
#         print("k_hat", result2['k_hat'], "k_selected ", k_selected)
#         back2 = True
#     # back1 = result1['backToSearch']
#     # back2 = result2['backToSearch']
#
#     # 更新当前最优
#     f_best_current = min(result1['f_best_current'], result2['f_best_current'], f_best_current)
#
#     if not (back1 or back2):
#         # 无需回退，更新区间集
#         L, f_min = update_boxes_lb(L, result1, result2, f_best_current, f_min)
#         if len(L) > 1:
#             L = [item for idx, item in enumerate(L) if idx != selected_order]
#         epoch += 1
#         print(f"epoch {epoch} | lbx: {result1['lbx']} {result2['lbx']} | L size: {len(L)}")
#         print(f"f_best_current: {f_best_current}, f_min: {f_min}")
#         return recursive_optimize(L, f_best_current, f_min, model, epsilon, epoch)
#     else:
#         # 回退
#         L = [{'x': x_initial, 'lbx': 0, 'f_best_in': min(result1['f_best_current'], result2['f_best_current']),
#               'k_hat': max(result1['k_hat'], result2['k_hat'],k_selected)}]
#         print(f"回退，更新 k_hat 为 {L[0]['k_hat']},回退，更新 f_best_in 为 {L[0]['f_best_in']}")
#         # 递归调用
#         return recursive_optimize(L, f_best_current, f_min, model, epsilon, 0)


# =====================================================================
def AdaLIPO(model, x_current, n: int, p=0.5):
    # Initialization
    t = 1
    alpha = 1e-2
    k_hat = 0
    b = 0
    bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
    X_1 = Uniform(bounds)
    nb_samples = 1

    points = X_1.reshape(1, -1)
    # values = np.array([model(torch.as_tensor(X_1).cuda().float()).cpu().detach().numpy()[0][0]])
    values = np.array([model(torch.as_tensor(X_1).float()).detach().numpy()[0][0]])
    minValue = values[0]

    def k(i):
        """
        Series of potential Lipschitz constants.
        """
        return (1 + alpha) ** i

    # Statistics
    stats = []
    a = 0
    # Main loop
    ratios = []
    i_hat = -np.inf
    while t < n:
        B_tp1 = Bernoulli(p)
        if B_tp1 == 1:
            # Exploration
            X_tp1 = Uniform(bounds)
            nb_samples += 1
            points = np.concatenate((points, X_tp1.reshape(1, -1)))
            # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
            value = np.array([model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]])
            # a = a+1
            t += 1
        else:
            # Exploitation
            while True:
                X_tp1 = Uniform(bounds)
                nb_samples += 1
                if LIPO_condition(X_tp1, values, k_hat, points) or t > n:
                    points = np.concatenate((points, X_tp1.reshape(1, -1)))
                    break
            # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
            value = np.array([model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]])
        if minValue > value:
            minValue = value
        values = np.concatenate((values, np.array([value])))
        for i in range(points.shape[0] - 1):
            ratios.append(
                np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
            )
        i_hat = max(int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha))), i_hat)
        k_hat = k(i_hat)
        # print(i_hat,k_hat)
        stats.append((np.max(values), nb_samples, k_hat))
    return k_hat, minValue


def AdaLIPO_P(model, x_current, x:np.array, n: int, window_slope=5, max_slope=800.0):
    """
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    p: probability of success for exploration/exploitation (float)
    fig_path: path to save the statistics figures (str)
    delta: confidence level for bounds (float)
    window_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
    max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
    """

    # Initialization
    global X_tp1
    t = 1
    alpha = 1e-1
    bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
    nb_samples = 1
    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)
    points = x.reshape(1, -1)
    # values = np.array([model(torch.as_tensor(X_1).cuda().float()).cpu().detach().numpy()[0][0]])
    values = np.array([model(torch.as_tensor(x).float()).detach().numpy()[0][0]])
    minValue = values[0]

    def k(i):
        """
        Series of potential Lipschitz constants.
        """
        return (1 + alpha) ** i

    # Statistics
    stats = []

    def p(t):
        """
        Probability of success for exploration/exploitation.
        """
        if t == 1:
            return 1
        else:
            return 1 / np.log(t)

    # Main loop
    ratios = []
    x_min, x_max = x_current[0], x_current[1]
    while t < n:
        B_tp1 = Bernoulli(p(t))
        if t<5:
            vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
            X_tp1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
            nb_samples += 1
            last_nb_samples[-1] = nb_samples
            points = np.concatenate((points, X_tp1.reshape(1, -1)))
            # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
            value = model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]
        else:
            if B_tp1 == 1:
                # Exploration
                X_tp1 = Uniform(bounds)
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
                points = np.concatenate((points, X_tp1.reshape(1, -1)))
                # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
                value = model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]
            else:
                # Exploitation
                while True:
                    X_tp1 = Uniform(bounds)
                    nb_samples += 1
                    last_nb_samples[-1] = nb_samples
                    if LIPO_condition(X_tp1, values, k_hat, points):
                        points = np.concatenate((points, X_tp1.reshape(1, -1)))
                        break
                    elif slope_stop_condition(last_nb_samples, max_slope):
                        print(
                            f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}."
                        )
                        # Output
                        return k_hat, minValue
                # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
                value = model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]
        values = np.concatenate((values, np.array([value])))
        for i in range(points.shape[0] - 1):
            ratios.append(
                np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
            )
        if minValue > value:
            minValue = value
        i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha)) + 2)
        k_hat = k(i_hat)
        # Statistical analysis
        stats.append((np.max(values), nb_samples, k_hat))
        t += 1
        # As `last_nb_samples` is a deque, we insert a 0 at the end of the list and increment this value by 1 for each point sampled instead of making a case distinction for the first sample and the others.
        last_nb_samples.append(0)
    return k_hat, minValue


def FeasibleSearch(model, x_current, n: int):
    bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
    t = 0
    x_min, x_max = x_current[0], x_current[1]
    minValue = np.inf
    X_tp1 =  Uniform(bounds)
    X = Uniform(bounds)
    while t < n:
        t=t+1
        if t<20:
            vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
            X_tp1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
        else:
            X_tp1 = Uniform(bounds)
        value = model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][0]
        if minValue > value:
            minValue = value
            X = X_tp1
    return minValue,X
import cma
import numpy as np
def convert_bounds(bounds_array):
    """
    将一个 2 × n 的 numpy.array 转为 CMA-ES 所需的 [lower_bounds, upper_bounds] 格式
    """
    assert bounds_array.shape[0] == 2, "bounds_array 应为 2 × n 的形状"
    lower_bounds = bounds_array[0, :].tolist()
    upper_bounds = bounds_array[1, :].tolist()
    return [lower_bounds, upper_bounds]
def rastrigin(x):
    return model(torch.as_tensor(x).float()).detach().numpy()[0][0]
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
x_initial = get_epsilon_neighborhood(image_tensor.view(-1, 14 * 14), epsilon=0.05)
epsilon = 1e-2  # 精度要求
fmin, f_best_current = interval_branch_and_bound(model, x_initial, epsilon)
print("结束", fmin, f_best_current)
