from collections import deque
import random
import numpy as np
import torch
from utils.LIPOUtil import Bernoulli, Uniform, LIPO_condition, slope_stop_condition

def generate_vertex(n):
    # 生成长度为 n 的随机 01 序列
    vertex = [random.randint(0, 1) for _ in range(n)]
    return vertex


def AdaLIPO_P(model, x_current, x:np.array, n: int, black_function,window_slope=5, max_slope=800.0,label = 0):
    global X_tp1
    t = 1
    alpha = 1e-1
    bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
    nb_samples = 1
    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)
    points = x.reshape(1, -1)
    # values = np.array([model(torch.as_tensor(X_1).cuda().float()).cpu().detach().numpy()[0][label]])
    values = np.array([black_function(x)])
    minValue = values[0]
    k_hat = 0
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
            vertex = generate_vertex(len(x_min))
            X_tp1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
            nb_samples += 1
            last_nb_samples[-1] = nb_samples
            points = np.concatenate((points, X_tp1.reshape(1, -1)))
            # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][label]
            value = black_function(X_tp1)
        else:
            if B_tp1 == 1:
                # Exploration
                X_tp1 = Uniform(bounds)
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
                points = np.concatenate((points, X_tp1.reshape(1, -1)))
                # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][label]
                value = black_function(X_tp1)
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
                        return k_hat, minValue
                # value = model(torch.as_tensor(X_tp1).cuda().float()).cpu().detach().numpy()[0][label]
                value = black_function(X_tp1)
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
