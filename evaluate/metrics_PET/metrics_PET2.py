'''
针对所有场景，根据场景特征分类后画热力图,
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 从metrics_PET.py中提取的PET计算方法
def calculate_speeds(x, y):
    speeds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / 0.5  # Speeds calculated between consecutive points
    speeds_padded = np.append(speeds, np.nan)  # Pad with NaN for the last point
    return speeds_padded


def evaluate_scene(scene_token_name, method, model=None):
    print(scene_token_name)
    if method == 'b_MPC':
        file_path = f"../../scenes_data/{scene_token_name}/{method}/Trj_result.csv"
    if model in ['llama3', 'llama3_no_memory', 'llama3_no_environment']:
        file_path = f"../../scenes_data/{scene_token_name}/{method}/{model}/Trj_result.csv"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return np.nan, np.nan

    data = pd.read_csv(file_path)

    # Identify rows where 'vehicle_id' is 'ego_vehicle' and 'front_vehicle' is 1
    ego_data = data[data['vehicle_id'] == 'ego_vehicle']
    front_data = data[data['front_vehicle'] == 1]

    # Ensure both datasets are sorted by time to align them by timestamp
    ego_data = ego_data.sort_values(by='timestamp')
    front_data = front_data.sort_values(by='timestamp')

    # Calculate speeds using the sorted ego vehicle data
    ego_data['speed'] = calculate_speeds(ego_data['x'], ego_data['y'])
    ego_data['speed_new'] = calculate_speeds(ego_data['x_new'], ego_data['y_new'])

    # Merge ego and front vehicle data based on timestamp to align them on the same time point
    merged_data = pd.merge(ego_data, front_data[['timestamp', 'x', 'y']],
                           on='timestamp', how='inner', suffixes=('_ego', '_front'))

    # Calculate distances using the positions of ego and front vehicles
    merged_data['distance'] = np.sqrt(
        (merged_data['x_ego'] - merged_data['x_front']) ** 2 + (merged_data['y_ego'] - merged_data['y_front']) ** 2)
    merged_data['distance_new'] = np.sqrt(
        (merged_data['x_new'] - merged_data['x_front']) ** 2 + (merged_data['y_new'] - merged_data['y_front']) ** 2)

    # Calculate PET using the distances and speeds
    merged_data['PET_origin'] = merged_data['distance'] / merged_data['speed']
    merged_data['PET_MPC'] = merged_data['distance_new'] / merged_data['speed_new']

    # Replace any infinite or NaN values with NaN and remove any NaN entries
    merged_data['PET_origin'].replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data['PET_MPC'].replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data.dropna(subset=['PET_origin', 'PET_MPC'], inplace=True)

    min_pet_value = merged_data['PET_origin'].min()
    min_pet_value_new = merged_data['PET_MPC'].min()
    print(f"Scene: {scene_token_name}, min_pet: {min_pet_value}, min_pet_new: {min_pet_value_new}")
    return min_pet_value, min_pet_value_new



# 定义计算方法
# method = 'b_MPC'
# method = "p_VLM_MPC1"
method = "b_LLM_MPC"
# model = "llama3"
model = "llama3_no_memory"
#model = "llama3_no_environment"



# 读取all_scenes.csv文件
all_scenes_df = pd.read_csv('../../scenes_info/all_scenes.csv')

# 获取../scenes_data/文件夹下的文件名集合
scenes_data_path = '../../scenes_data/'
set1 = set(os.listdir(scenes_data_path))

# 筛选出文件名在set1中的场景
filtered_scenes_df = all_scenes_df #[all_scenes_df['scene_token'].isin(set1)]

# 计算每个场景的最小PET值
filtered_scenes_df[['min_pet', 'min_pet_new']] = filtered_scenes_df['scene_token'].apply(lambda x: pd.Series(evaluate_scene(x, method, model)))

# 计算每种组合的最小PET值平均值
combinations_pet_origin = filtered_scenes_df.groupby(['rain', 'intersection', 'night'])['min_pet'].min().reset_index()
combinations_pet_new = filtered_scenes_df.groupby(['rain', 'intersection', 'night'])['min_pet_new'].min().reset_index()

print('combinations_pet_new', combinations_pet_new)


def plot_pet_matrix(combinations_pet, title, min_pet, max_pet, filename):
    # 定义颜色映射，其中1.5对应白色
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=5)
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # 分别绘制rain=0和rain=1的情况
    for idx, rain_value in enumerate([0, 1]):
        ax = axes[idx]
        subset = combinations_pet[combinations_pet['rain'] == rain_value]

        # 创建一个空白的2x2矩阵
        pet_matrix = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])

        # 填充矩阵
        for _, row in subset.iterrows():
            pet_matrix.at[row['night'], row['intersection']] = row['min_pet'] if 'Original' in title else row['min_pet_new']

        # 绘制热力图
        cax = ax.imshow(pet_matrix, cmap='coolwarm_r', norm=norm)

        # 设置刻度和标签
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
        ax.set_xlabel('Intersection')
        ax.set_ylabel('Night')
        ax.set_title(f'rain={rain_value}')

        # 在每个格子上显示指标
        for (i, j), val in np.ndenumerate(pet_matrix.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
    # # 添加颜色条
    # cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)
    # cbar.ax.tick_params(labelsize=14)  # 设置颜色条刻度标签的字体大小
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    #plt.show()

# 确定颜色条的范围
max_pet = max(combinations_pet_origin['min_pet'].max(), combinations_pet_new['min_pet_new'].max())
min_pet = min(combinations_pet_origin['min_pet'].min(), combinations_pet_new['min_pet_new'].min())

# 绘制原始轨迹的平均最小PET值
plot_pet_matrix(combinations_pet_origin, 'Original Trajectory', min_pet, max_pet, 'metrics_PET_original.png')

# 绘制新轨迹的平均最小PET值
plot_pet_matrix(combinations_pet_new, 'New Trajectory', min_pet, max_pet, f'metrics_PET_{method}_{model}.png')