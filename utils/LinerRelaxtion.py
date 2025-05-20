import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
import random

# 根据f,返回一组约束
def LinearRelaxation(x, f, dfdx):
    # 一维情形，x取 x[0], x[1]
    res_min = minimize_scalar(dfdx, bounds=(x[0], x[1]), method='bounded')
    res_max = minimize_scalar(lambda x: -dfdx(x), bounds=(x[0], x[1]), method='bounded')
    A_ub =[[res_min.fun,-1 ],[-res_max.fun,-1 ]]
    b_ub =[res_min.fun * x[0] - f(x[0]), -res_max.fun * x[1] - f(x[1])]
    return {
        "A_ub": A_ub,
        "b_ub": b_ub
    }
# 根据g,返回一组约束
def LinearRelaxationForG(x, g, _g):
    # 一维情形，x取 x[0], x[1]
    res_min = minimize_scalar(_g, bounds=(x[0], x[1]), method='bounded')
    res_max = minimize_scalar(lambda x: -_g(x), bounds=(x[0], x[1]), method='bounded')
    print(f"Gradient range: [{res_min.fun}, {-res_max.fun}]")
    A_ub =[[res_min.fun ],[-res_max.fun ]]
    b_ub =[res_min.fun * x[0] - g(x[0]), -res_max.fun * x[1] - g(x[1])]
    return {
        "A_ub": A_ub,
        "b_ub": b_ub
    }
def PolytopeHull(x, f, g,dfdx,dgdx):

    # 线性松弛
    # gl = LinearRelaxation(x, g, RT)
    #
    # n = len(x)  # 假设 x 是一个向量，每个元素代表一个变量
    # for i in range(n):
    #     # 对每个变量进行收缩操作：最小化和最大化
    #     # 求解：min xi s.t. gl(x) <= 0

    # 计算目标函数的线性松弛
    fl = LinearRelaxation(x, f, dfdx)
    A_ub = fl['A_ub']
    b_ub = fl['b_ub']
    x_range = x
    plot_lines(f, A_ub, b_ub, x_range)
    # 合并fl和gl
    # 计算下界
    result = linprog([0,1], A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=np.array([x,(None, None)]), method='highs')
    lbx = result.fun
    return {'x':x, 'lbx' :lbx, 'success':result.success}

def plot_lines(f, A_ub, b_ub, x_range):
    # 定义 x 轴范围
    x = np.linspace(x_range[0], x_range[1], 400)

    # 计算 y 值
    y1 = A_ub[0][0] * x - b_ub[0]
    y2 = A_ub[1][0] * x - b_ub[1]
    y3 = f(x)
    # 绘制直线
    plt.plot(x, y1, label=f"y = {A_ub[0][0]:.3f} * x - ({b_ub[0]:.3f})", color='b')
    plt.plot(x, y2, label=f"y = {A_ub[1][0]:.3f} * x - ({b_ub[1]:.3f})", color='b')
    plt.plot(x, y3, label="f(x)=3*x**3 - 2*(x + 1/2)**2 + 2*x + 1", color='r')
    # 设置图例和标签
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    # plt.axhline(0, color='black', linewidth=0.5)
    # plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图像
    plt.show()
def FeasibleSearch(x, f, g, epsilon):
    x_best_current = x[0]
    f_best_current = f(x[0])
    for i in range(10):
        xrandom = random.uniform(x[0], x[1])
        frandom = f(xrandom)
        if(frandom < f_best_current):
            x_best_current = xrandom
            f_best_current = frandom
    return {'x_best_current':x_best_current, 'f_best_current':f_best_current}
    # res_min = minimize_scalar(f, bounds=(x[0], x[1]), method='bounded')
    # return {'x_best_current':res_min.x,
    # 'f_best_current':res_min.fun}
def contract_and_bound(x, f, g, f_best_current, epsilon):
    # TODO 合并约束
    global result
    dfdx = lambda x: 9 * x ** 2 - 4 * x
    dgdx = lambda x: 1
    x_lbx= PolytopeHull(x,f,g,dfdx,dgdx)
    x, lbx = x_lbx['x'],x_lbx['lbx']
    if x_lbx['success']:
        result = FeasibleSearch(x,f,g,epsilon)
    return {
        'x':x,
        'lbx':lbx,
        'x_best_current': result['x_best_current'],
        'f_best_current': result['f_best_current'],
    }
def best_box(L):
    lb_min = np.inf
    selected_order = 0
    for (i,d) in enumerate(L):
        if d['lbx'] < lb_min:
            selected_order = i
            lb_min = d['lbx']
    return selected_order
def bisect(x):
    """
    对box的某一个维度进行分割

    这里x是一维的，对区间进行二分。
    x: 当前区间的解
    """
    # 选择具有最大宽度的维度，并通过中间点对其进行二分。 也可以采用启发式算法
    start, end = x
    # 计算中间点
    mid = (start + end) / 2

    # 返回两个子区间
    x1 = (start, mid)
    x2 = (mid, end)

    return x1, x2
def update_boxes_lb(L, result1, result2,f_min):

    L.append({'x':result1['x'], 'lbx':result1['lbx']})
    L.append({'x':result2['x'], 'lbx':result2['lbx']})
    # 这里我们只保留那些下界大于目标值 f_min 的区间
    new_L = []
    for d in L:
        if(d['lbx'] < f_min):
            new_L.append(d)
    return new_L
def interval_branch_and_bound(f, g, x, epsilon):
    f_min = -np.inf  # 初始化下界最小值
    f_best_current = np.inf  # 当前最优解
    lbx = -np.inf
    L = [{'x':x,'lbx':lbx}]  # 初始化区间列表，包含初始解和下界
    while L and f_best_current - f_min > epsilon:
        # 选择最优区间
        selected_order = best_box(L)
        x_selected = L[selected_order]['x']
        L.remove(L[selected_order])
        f_min = x_selected['lbx']
        # 对当前区间x_selected进行二分
        x1, x2 = bisect(x_selected)
        #
        result1 = contract_and_bound(x1, f, g, f_best_current, epsilon)
        result2 = contract_and_bound(x2, f, g, f_best_current, epsilon)
        # 更新最小值和待处理区间
        f_min = min(result1['f_best_current'],result2['f_best_current'])
        print("lbx",result1['lbx']," ", result2['lbx'], "L_size", len(L))
        print("f_best_current",f_best_current,"f_min",f_min)
        L = update_boxes_lb(L, result1, result2,f_min)

    return f_min, L  # 返回最优解和剩余区间

x_initial = np.array([0, 1])  # 初始解
f = lambda x: 3*x**3 - 2*(x + 1/2)**2 + 2*x + 1
g = lambda x: x
epsilon = 1e-3  # 精度要求
fmin, L = interval_branch_and_bound(f, g, x_initial, epsilon)
print(f"最优解：{fmin}, 待处理区间：{L}")