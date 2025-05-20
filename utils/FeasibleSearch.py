# from collections import deque
# import random
# import numpy as np
# import torch
# from utils.LIPOUtil import Bernoulli, Uniform, LIPO_condition, slope_stop_condition
#
# def FeasibleSearch(model, x_current, n: int,label):
#     bounds = np.array([[x_min, x_max] for x_min, x_max in zip(x_current[0, :], x_current[1, :])])
#     t = 0
#     x_min, x_max = x_current[0], x_current[1]
#     minValue = np.inf
#     X_tp1 =  Uniform(bounds)
#     X = Uniform(bounds)
#     while t < n:
#         t=t+1
#         if t<20:
#             vertex, opposite_vertex = generate_vertex_and_opposite(len(x_min))
#             X_tp1 = np.array([x_min[i] if v == 0 else x_max[i] for i, v in enumerate(vertex)])
#         else:
#             X_tp1 = Uniform(bounds)
#         value = model(torch.as_tensor(X_tp1).float()).detach().numpy()[0][label]
#         if minValue > value:
#             minValue = value
#             X = X_tp1
#     return minValue,X