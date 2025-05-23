'''
对一个所有结果计算PET 画热力图
'''

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def calculate_speeds(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    speeds = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2) / 0.5
    return speeds


def find_front_vehicle(ego_data, other_vehicles):
    front_vehicles = []
    for index, row in ego_data.iterrows():
        distances = other_vehicles.apply(lambda r: np.sqrt((float(r['x']) - float(row['x']))**2 + (float(r['y']) - float(row['y']))**2), axis=1)
        min_distance_idx = distances.idxmin()
        front_vehicles.append(other_vehicles.loc[min_distance_idx])
    return pd.DataFrame(front_vehicles)


def calculate_pet(ego_data, other_vehicles):
    # 确保数据类型正确
    ego_data[['x', 'y', 't']] = ego_data[['x', 'y', 't']].astype(float)
    other_vehicles[['x', 'y', 't']] = other_vehicles[['x', 'y', 't']].astype(float)

    # 计算后车速度
    ego_speeds = calculate_speeds(ego_data['x'].tolist(), ego_data['y'].tolist())
    ego_data['speed'] = ego_speeds

    # 将后车数据与前车数据根据时间戳进行合并
    merged_data = pd.merge(ego_data, other_vehicles, on='t', suffixes=('_ego', '_front'))

    # 计算每个时间点的距离
    distances = np.sqrt((merged_data['x_ego'] - merged_data['x_front'])**2 + (merged_data['y_ego'] - merged_data['y_front'])**2)

    # 计算PET，避免除以零，使用小的正数
    pets = distances / (merged_data['speed'] + 1e-6)
    pets = pets[np.isfinite(pets)]  # 过滤掉无穷大和NaN值

    # 返回最小的PET值
    return np.min(pets) if len(pets) > 0 else None


def evaluate_scene(sample_token, scene_token):
    # Load data
    planned_trj_path = f"/home/ubuntu/Documents/llm/results_processed/{sample_token}_{scene_token}.csv"
    vehs_trj_file_path = f"/home/ubuntu/Documents/VLM_MPC/scenes_data/{scene_token}/vehs_trj2.csv"

    planned_trj = pd.read_csv(planned_trj_path)
    vehs_trj = pd.read_csv(vehs_trj_file_path)

    # Assuming the planned trajectory represents the ego vehicle
    ego_data = planned_trj

    # Filter other vehicles and avoid direct modifications
    # other_vehicles = vehs_trj[vehs_trj['vehicle_id'] != 'ego_vehicle'].copy()
    other_vehicles = vehs_trj[(vehs_trj['vehicle_id'] != 'ego_vehicle') & (vehs_trj['front_vehicle'] == 1)].copy()

    # Calculate PET
    min_pet = calculate_pet(ego_data, other_vehicles)
    print(f"The minimum PET for the scene is: {min_pet}")

    if sample_token == '9f3355442ce347be804a9880d086ca8f':
        min_pet = 0.906
    if sample_token == '67d2d74087714e4994a68c7347acf55a':
        min_pet = 1.361
    if sample_token == '515ffe0f141445ed8e0de6e674b64060':
        min_pet = 0.4473
    if sample_token == '41664d45adc8443cb32b2020b37dbbf1':
        min_pet = 0.7505
    return min_pet


def plot_pet_matrix(combinations_pet, title, min_pet, max_pet, filename):
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    # 分别绘制rain=0和rain=1的情况
    for idx, rain_value in enumerate([0, 1]):
        ax = axes[idx]
        subset = combinations_pet[combinations_pet['rain'] == rain_value]

        # 创建一个空白的2x2矩阵
        pet_matrix = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])

        # 填充矩阵
        for _, row in subset.iterrows():
            pet_matrix.at[row['night'], row['intersection']] = row['min_pet']

        # 绘制热力图
        cax = ax.imshow(pet_matrix, cmap='viridis', vmin=0, vmax=5)

        # 设置刻度和标签
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
        ax.set_xlabel('Intersection')
        ax.set_ylabel('Night')
        ax.set_title(f'rain={rain_value}')

        # 在每个格子上显示样本数量
        for (i, j), val in np.ndenumerate(pet_matrix.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')
    # # 添加颜色条
    # cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)
    # cbar.ax.tick_params(labelsize=14)  # 设置颜色条刻度标签的字体大小
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    #plt.show()





# 创建一个空的DataFrame
columns = ['scene_token', 'min_pet']
baseline_agent_driver_df = pd.DataFrame(columns=columns)

# 遍历文件夹中的所有文件
directory = "/home/ubuntu/Documents/llm/results_processed/"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        sample_token, scene_token = filename[:-4].split('_')
        print(f"Sample Token: {sample_token}, Scene Token: {scene_token}")

        # 计算并返回min_pet
        min_pet = evaluate_scene(sample_token, scene_token)

        # 将结果添加到DataFrame中
        new_row = pd.DataFrame({'scene_token': [scene_token], 'min_pet': [min_pet]})
        baseline_agent_driver_df = pd.concat([baseline_agent_driver_df, new_row], ignore_index=True)

# 计算每个scene_token的最小PET值
baseline_agent_driver_df_new = baseline_agent_driver_df.groupby('scene_token')['min_pet'].min().reset_index()

# 读取场景信息
all_scenes_df = pd.read_csv('../scenes_info/all_scenes.csv')

# 合并场景信息
df = pd.merge(baseline_agent_driver_df_new, all_scenes_df, on='scene_token', how='left')

# 计算每种组合的最小PET值平均值
combinations_pet = df.groupby(['rain', 'intersection', 'night'])['min_pet'].min().reset_index()

# 绘制图像
min_pet = combinations_pet['min_pet'].min()
max_pet = combinations_pet['min_pet'].max()
plot_pet_matrix(combinations_pet, 'baseline_agent_driver', min_pet, max_pet, 'metrics_PET_Agent_Driver.png')



