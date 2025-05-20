import random
from collections import deque

import torch
from scipy.optimize import linprog

from network.DNN1 import SimpleNN14
from utils.LIPOUtil import *

# from pythonProject2.network.SimpleNN import SimpleNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)  # 输入层：784 -> 128
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)  # 隐藏层：128 -> 64
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(64, 10)  # 输出层：64 -> 10（10类）
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # 展平输入
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.fc3(x)
#         return x


model = SimpleNN14().to(device)
model.load_state_dict(torch.load("../network/DNN1.pth"))
model.eval()  # 进入推理模式
state_dict = model.state_dict()

from PIL import Image
from torchvision import transforms

# 读取图像文件
image = Image.open('../img[0].png')
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
    # start = np.array(start)
    # end = np.array(end)
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
def LinearRelaxation(x, model, k):
    A_ub,b_ub = [],[]
    x_min, x_max = x[0],x[1]
    for i in range(2):
        A_ub.append([-k,-1])
        A_ub.append([k,-1])
        vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
        x0 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
        x1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(opposite_vertex)])
        sum_x0 ,sum_x1 = sum(x0), sum(x1)
        fx0= model(torch.as_tensor(x0).cuda().float()).cpu().detach().numpy()[0][0]
        fx1 = model(torch.as_tensor(x1).cuda().float()).cpu().detach().numpy()[0][0]
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



def PolytopeHull(x, model, k):
    # 计算目标函数的线性松弛
    fl = LinearRelaxation(x, model, k)
    A_ub = fl['A_ub']
    b_ub = fl['b_ub']
    # 计算下界
    result = linprog([0,1], A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=np.array([(sum(x[0]),sum(x[1])),(None,None)]), method='highs')
    lbx = result.fun
    return {'x':x, 'lbx' :lbx, 'success':result.success}
def contract_and_bound(x, model, f_best_current, epsilon,k_selected,f_best_in,f_min):
    k_hat, value = AdaLIPO_P(model, x_initial, 1000)
    backToSearch = False
    # print(k_hat, value)
    x_lbx= PolytopeHull(x,model,k_hat)
    x, lbx = x_lbx['x'],x_lbx['lbx']
    if not x_lbx['success']:
        raise ValueError("线性规划求解失败")
    if k_hat > k_selected:
        print("k_hat",k_hat, "k_selected ",k_selected)
        backToSearch = True
    elif f_best_in < value:
        print("f_best_in",k_hat, "value ",k_selected)
        backToSearch = True
    elif lbx > value:
        print("lbx", lbx, "value ", value)
        backToSearch = True
    elif  lbx < f_min:
        print("lbx",lbx, " f_min", f_min)
        backToSearch = True
        # raise ValueError("lb没有变大")
    return {
        'x':x,
        'lbx':lbx,
        'f_best_current': value,
        'k_hat':k_hat,
        'backToSearch':backToSearch
    }
def update_boxes_lb(L, result1, result2,f_best_current,f_min):

    L.append({'x':result1['x'], 'lbx':result1['lbx'],'k_hat':result1['k_hat'],'f_best_in':result1['f_best_current']})
    L.append({'x':result2['x'], 'lbx':result2['lbx'],'k_hat':result2['k_hat'],'f_best_in':result2['f_best_current']})
    # 这里我们只保留那些下界大于目标值 f_min 的区间
    new_L = []
    for d in L:
        if(d['lbx'] < f_best_current):
            new_L.append(d)
            f_min = max(d['lbx'], f_min)
    return new_L,f_min
def interval_branch_and_bound(model, x, epsilon):
    global result1, result2
    f_min = -np.inf  # 初始化下界最小值
    f_best_current = np.inf  # 当前最优解
    lbx = -np.inf
    k_hat = np.inf
    L = [{'x': x, 'lbx': lbx,'k_hat':k_hat,'f_best_in':f_best_current}]  # 初始化区间列表，包含初始解和下界
    epoch = 0
    # while L and f_best_current - f_min > epsilon:
    while f_best_current - f_min > epsilon:
        # 选择最优区间
        selected_order = best_box(L)
        x_selected = L[selected_order]['x']
        k_selected = L[selected_order]['k_hat']
        f_min = L[selected_order]['lbx']
        f_best_in = L[selected_order]['f_best_in']
        # 对当前区间x_selected进行二分
        x1, x2 = bisect(x_selected)
        result1 = contract_and_bound(x1, model, f_best_current, epsilon,k_selected,f_best_in,f_min)
        backToSearch1 = result1['backToSearch']
        backToSearch2 = True
        result2 = contract_and_bound(x2, model, f_best_current, epsilon,k_selected,f_best_in,f_min)
        backToSearch2 = result2['backToSearch']
        f_best_current = min(result1['f_best_current'], result2['f_best_current'],f_best_current)
        if not (backToSearch1 or backToSearch2):
            L, f_min = update_boxes_lb(L, result1, result2, f_best_current,f_min)
            if len(L)>1:
                L = [item for item in L if not np.array_equal(item, L[selected_order])]
            epoch+=1
            print("epoch",epoch,"lbx",result1['lbx']," ", result2['lbx'], "L_size", len(L))
            print("f_best_current",f_best_current,"f_min",f_min)
        else:
            L[selected_order]['k_hat'] = result1['k_hat']
    return f_min, f_best_current  # 返回最优解和剩余区间

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
    values = np.array([model(torch.as_tensor(X_1).cuda().float()).cpu().detach().numpy()[0][0]])
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
    i_hat=-np.inf
    while t < n:
        B_tp1 = Bernoulli(p)
        if B_tp1 == 1:
            # Exploration
            X_tp1 = Uniform(bounds)
            nb_samples += 1
            points = np.concatenate((points, X_tp1.reshape(1, -1)))
            value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
            # a = a+1
            t += 1
        else:
            # Exploitation
            while True:
                X_tp1 = Uniform(bounds)
                nb_samples += 1
                if LIPO_condition(X_tp1, values, k_hat, points) or t>n:
                    points = np.concatenate((points, X_tp1.reshape(1, -1)))
                    break
            value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
        if minValue>value:
            minValue = value
        values = np.concatenate((values, np.array([value])))
        for i in range(points.shape[0] - 1):
            ratios.append(
                np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
            )
        i_hat = max(int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha))),i_hat)
        k_hat = k(i_hat)
        # print(i_hat,k_hat)
        stats.append((np.max(values), nb_samples, k_hat))
    return k_hat,minValue


def AdaLIPO_P(model,x_current, n: int, window_slope=5, max_slope=800.0):
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
    t = 1
    alpha = 1e-2
    k_hat = 0
    bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
    X_1 = Uniform(bounds)
    nb_samples = 1

    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)

    points = X_1.reshape(1, -1)
    values = np.array([model(torch.as_tensor(X_1).cuda().float()).cpu().detach().numpy()[0][0]])
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
    while t < n:
        B_tp1 = Bernoulli(p(t))
        if B_tp1 == 1:
            # Exploration
            X_tp1 = Uniform(bounds)
            nb_samples += 1
            last_nb_samples[-1] = nb_samples
            points = np.concatenate((points, X_tp1.reshape(1, -1)))
            value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
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
            value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][0]
        values = np.concatenate((values, np.array([value])))
        for i in range(points.shape[0] - 1):
            ratios.append(
                np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
            )
        if minValue>value:
            minValue = value
        i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha))+2)
        k_hat = k(i_hat)
        # Statistical analysis
        stats.append((np.max(values), nb_samples, k_hat))
        t += 1
        # As `last_nb_samples` is a deque, we insert a 0 at the end of the list and increment this value by 1 for each point sampled instead of making a case distinction for the first sample and the others.
        last_nb_samples.append(0)
    return k_hat, minValue


# ===================================================================
x_initial = get_epsilon_neighborhood(image_tensor.view(-1, 14 * 14), epsilon=0.1)
# print(x_initial)
# k_hat, value = AdaLIPO(model, x_initial, 10000)
# # f = lambda x: 3*x**3 - 2*(x + 1/2)**2 + 2*x + 1
# g = lambda x: x
epsilon = 1e-2  # 精度要求
fmin, f_best_current = interval_branch_and_bound(model, x_initial, epsilon)
print("结束",fmin, f_best_current)