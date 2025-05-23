import numpy as np
from cvxpy import *


def mpc_fun( v_ego, lead_info, N, dt, Q, R, Q_h, tau, desired_speed, desired_headway):
    s0 = 4.2
    min_headway = 1
    slack_penalty = 1e5  # 软约束的惩罚系数，确保优化函数依然会尽量满足原来的约束

    d = Variable((N+1, 1))
    v = Variable((N+1, 1))
    u = Variable((N, 1))  # 加速度

    a_sigma = Variable((N, 1), nonneg=True)  # 加速度软约束的松弛变量
    d_sigma = Variable((N, 1), nonneg=True)  # 安全距离软约束的松弛变量

    constraints = [d[0] == 0, v[0] == v_ego]
    for i in range(N):
        constraints += [
            d[i+1] == d[i] + dt * v[i],
            v[i+1] == v[i] + dt * (u[i] / tau),  # 考虑引擎滞后的加速度模型
            v[i] <= desired_speed+ a_sigma[i],
            v[i] >= 0,
            u[i] <= 2.5 + a_sigma[i],
            u[i] >= -3 - a_sigma[i],
        ]

    cost = sum([Q * (v[i] - desired_speed) ** 2 + R * u[i] ** 2 + slack_penalty * a_sigma[i] for i in range(N)])

    if lead_info:
        d0, lead_v = lead_info
        for i in range(N):
            # 计算期望距离
            expected_distance = d0 + lead_v * i * dt - d[i] - desired_headway * v[i] - s0
            cost += Q_h * (expected_distance)**2

            # 安全约束：headway 不小于 min_headway
            constraints += [
                                d0 + lead_v * i * dt - d[i] - min_headway * v[i] - s0 >= 0
            ]
            #cost += 1e6 * d_sigma[i]

    prob = Problem(Minimize(cost), constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise ValueError("MPC optimization problem did not solve successfully.")

    return d.value, v.value, u.value
