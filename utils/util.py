import random
from collections import deque

import numpy as np
import torch


def generate_vertex_and_opposite(n):
    # 生成长度为 n 的随机 01 序列
    vertex = [random.randint(0, 1) for _ in range(n)]
    # 生成相对的顶点：0 变 1，1 变 0
    opposite_vertex = [1 - bit for bit in vertex]
    return vertex, opposite_vertex
def is_ancestor(node, maybeAncestorNode) -> bool:
    while node.parent is not None:
        node = node.parent
        if node is maybeAncestorNode:
            return True
    return False

def print_all_nodes(root,logging):
    queue = deque([root])
    node_number = 1
    while queue:
        node = queue.popleft()
        logging.info(f"Node {node_number}: k={node.k}, f_best={node.f_best}, f_min={node.f_min}")
        node_number += 1
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

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

# def bisect(x):
#     # 选择具有最大宽度的维度，并通过中间点对其进行二分。 也可以采用启发式算法
#     start, end = x[0], x[1]
#     # 计算每个维度的宽度
#     widths = end - start
#     # 找到最大宽度的维度索引
#     max_dim = np.argmax(widths)
#     # 计算该维度的中点
#     mid = (start[max_dim] + end[max_dim]) / 2
#     # 生成两个新区间
#     mid1 = start.copy()
#     mid2 = end.copy()
#     mid1[max_dim] = mid
#     mid2[max_dim] = mid
#     x1 = np.array([start.tolist(), mid2.tolist()])
#     x2 = np.array([mid1.tolist(), end.tolist()])
#     return x1, x2,max_dim
