import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 定义自行车二轮模型
class BicycleModel:
    def __init__(self, dt, L):
        self.dt = dt  # 时间步长
        self.L = L  # 轴距

    def update(self, x, u):
        # x = [x, y, theta, v]
        # u = [a, delta]
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + x[3] * np.cos(x[2]) * self.dt
        x_next[1] = x[1] + x[3] * np.sin(x[2]) * self.dt
        x_next[2] = x[2] + x[3] / self.L * np.tan(u[1]) * self.dt
        x_next[3] = x[3] + u[0] * self.dt
        return x_next

# 拟合直线函数
def fit_line(centerline):
    x = centerline[:, 0]
    y = centerline[:, 1]
    coeffs = np.polyfit(x, y, 1)  # 拟合直线
    return coeffs

# 计算点到直线的距离
def point_to_line_dist(x, y, coeffs):
    return np.abs(coeffs[0] * x - y + coeffs[1]) / np.sqrt(coeffs[0]**2 + 1)

# 定义MPC优化问题
def mpc_fun(model, x0, coeffs, N, Q, R, desired_speed):
    N =10
    def cost_fn(u):
        cost = 0.0
        x = x0.copy()
        for i in range(N):
            x = model.update(x, u[i * 2:i * 2 + 2])
            dist_to_line = point_to_line_dist(x[0], x[1], coeffs)
            cost += Q[0] * dist_to_line**2
            cost += Q[1] * (x[3] - desired_speed)**2
            cost += R[0] * u[i * 2]**2 + R[1] * u[i * 2 + 1]**2
        return cost

    u0 = np.zeros(2 * N)  # 初始控制输入
    bounds = [(-1, 1), (-np.pi / 4, np.pi / 4)] * N  # 控制输入约束
    result = minimize(cost_fn, u0, bounds=bounds, options={'disp': True, 'maxiter': 100}, method='SLSQP')
    return result.x

# 仿真参数
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
nusc_map = NuScenesMap(dataroot=dataroot, map_name='boston-seaport')

dt = 0.5  # 时间步长

scene_token_name = '0c12123d57594f4d81d32fd4aa89fb1f'

# 从params中提取MPC参数
data = pd.read_csv(f"../scenes_data/{scene_token_name}/MPC_calibration/best_MPC_parameter.csv", header=None)
N, Q0, R0, Q_h, desired_speed, desired_headway = data.iloc[-1]
Q = [Q0, 1]
R = [R0, 1]

# 读取车辆轨迹数据
df = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")
ego_vehicle_id = 'ego_vehicle'
ego_df = df[df['vehicle_id'] == ego_vehicle_id]

# 初始化轨迹
x_new = [ego_df.iloc[0]['x']]
y_new = [ego_df.iloc[0]['y']]
theta_new = [np.arctan2(ego_df.iloc[1]['y'] - ego_df.iloc[0]['y'], ego_df.iloc[1]['x'] - ego_df.iloc[0]['x'])]
vx = np.sqrt((ego_df.iloc[1]['x'] - ego_df.iloc[0]['x']) ** 2 + (ego_df.iloc[1]['y'] - ego_df.iloc[0]['y']) ** 2) / dt

# 读取中线CSV文件
lane_centerlines_df = pd.read_csv(f"../scenes_data/{scene_token_name}/lane_centerlines.csv")
lane_centerlines = lane_centerlines_df.values

# 拟合中心线为直线
coeffs = fit_line(lane_centerlines)

model = BicycleModel(dt, 2.0)

for i in range(1, len(ego_df)):
    x0 = np.array([x_new[-1], y_new[-1], theta_new[-1], vx])
    u_opt = mpc_fun(model, x0, coeffs, N, Q, R, desired_speed)
    x0 = model.update(x0, u_opt[:2])

    x_new.append(x0[0])
    y_new.append(x0[1])
    theta_new.append(x0[2])
    vx = x0[3]

ego_df = df[df['vehicle_id'] == 'ego_vehicle'].copy()
ego_df.loc[:, 'x_new'] = pd.Series(x_new[:len(ego_df)], index=ego_df.index)
ego_df.loc[:, 'y_new'] = pd.Series(y_new[:len(ego_df)], index=ego_df.index)
ego_df.loc[:, 'theta_new'] = pd.Series(theta_new[:len(ego_df)], index=ego_df.index)

df.loc[df['vehicle_id'] == ego_vehicle_id, 'x_new'] = np.nan
df.loc[df['vehicle_id'] == ego_vehicle_id, 'y_new'] = np.nan
df.loc[df['vehicle_id'] == ego_vehicle_id, 'theta_new'] = np.nan
df.update(ego_df[['vehicle_id', 'timestamp', 'x_new', 'y_new', 'theta_new']])
df.to_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv", index=False)


import matplotlib.pyplot as plt
# 从文件中读取轨迹数据并绘制
df_updated = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")
ego_df_updated = df_updated[df_updated['vehicle_id'] == 'ego_vehicle']
# 绘制轨迹
plt.figure(figsize=(10, 8))
plt.plot(lane_centerlines_df['x'], lane_centerlines_df['y'], 'g--', label='referenced centerline')
plt.plot(ego_df['x'], ego_df['y'], '.', label='original trajectory')
plt.plot(ego_df_updated['x_new'], ego_df_updated['y_new'], 'b-', label='MPC result')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()