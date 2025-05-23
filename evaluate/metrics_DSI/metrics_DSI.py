'''
计算单个场景的DSI，然后保存
'''

import pandas as pd
import numpy as np
from itertools import combinations
import os
import matplotlib.pyplot as plt



def calculate_DSI(data_dir, scene_token_name):
    # 调整参数
    M = 1500.0
    K = 1.0  # 增大K以放大场的影响
    k = 2     # 减小衰减系数以减慢场强的衰减速度

    # 读取CSV文件
    data = pd.read_csv(f"{data_dir}/{scene_token_name}/vehs_trj.csv")

    # 计算速度和加速度
    data['delta_x'] = data.groupby('vehicle_id')['x'].diff()
    data['delta_y'] = data.groupby('vehicle_id')['y'].diff()
    data['delta_t'] = data.groupby('vehicle_id')['timestamp'].diff()
    data['velocity'] = np.sqrt(data['delta_x']**2 + data['delta_y']**2) / data['delta_t']

    # 准备数据表来计算潜在场和动力场
    results = pd.DataFrame()

    for time, group in data.groupby('timestamp'):
        vehicles_data = group[['vehicle_id', 'x', 'y', 'velocity']].dropna()
        total_dsi = 0
        x_ego = y_ego = None
        for (id1, x1, y1, v1), (id2, x2, y2, v2) in combinations(vehicles_data.values, 2):
            if 'ego_vehicle' in [id1, id2]:
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance > 0:
                    potential_field = K * M * M / distance ** k
                    kinetic_field = K * M * M * np.abs(v1 - v2) / distance ** k
                    total_safety_index = potential_field + kinetic_field
                    total_dsi += total_safety_index
                    if id1 == 'ego_vehicle':
                        x_ego, y_ego = x1, y1
                    else:
                        x_ego, y_ego = x2, y2
        if x_ego is not None and y_ego is not None:
            results = pd.concat([results, pd.DataFrame({'timestamp': [time], 'x': [x_ego], 'y': [y_ego], 'total_dsi': [total_dsi]})], ignore_index=True)

    return results

def plot_DSI(results, save_path):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(results['x'], results['y'], c=results['total_dsi'], cmap='viridis', vmin=results['total_dsi'].min(), vmax=results['total_dsi'].max(), marker='o')
    plt.colorbar(sc, label='Total DSI')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Ego Vehicle Trajectory Colored by Total DSI')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    data_dir = '../../scenes_data/'
    scene_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for scene_token_name in scene_folders:
        print(f'process {scene_token_name}')
        if os.path.exists(f'{data_dir}/{scene_token_name}/vehs_trj.csv'):
            results = calculate_DSI(data_dir, scene_token_name)
            results.to_csv(f'{data_dir}/{scene_token_name}/DSI.csv', index=False)
            plot_DSI(results, f'{data_dir}/{scene_token_name}/DSI.png')
