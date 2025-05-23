'''
对一个所有结果计算smoothness 画热力图
'''

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def calculate_smoothness(x, y):
    """ Calculate the RMS of acceleration for given x, y coordinates and timestamps. """
    dt = 0.5
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ddx = np.diff(dx, prepend=dx[0]) / dt
    ddy = np.diff(dy, prepend=dy[0]) / dt
    acceleration = np.sqrt(ddx ** 2 + ddy ** 2)
    rms_value = np.sqrt(np.mean(acceleration ** 2))
    return rms_value


def evaluate_scene(sample_token, scene_token):
    """ Evaluate the smoothness of the trajectory for a given scene. """
    # Load the trajectory data for the specified scene
    planned_trj_path = f"/home/ubuntu/Documents/llm/results_processed/{sample_token}_{scene_token}.csv"
    planned_trj = pd.read_csv(planned_trj_path)

    # Assume that the CSV file contains columns 'x', 'y', and 'timestamp'
    x = planned_trj['x']
    y = planned_trj['y']

    # Calculate smoothness
    smoothness = calculate_smoothness(x, y)
    return smoothness


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
        cax = ax.imshow(pet_matrix, cmap='viridis', vmin=0.25, vmax=0.8)
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

path2 = "/home/ubuntu/Documents/VLM_MPC/scenes_data"
# 获取路径2下所有子文件夹的名称
folders_in_path2 = {folder for folder in os.listdir(path2) if os.path.isdir(os.path.join(path2, folder))}


# 遍历文件夹中的所有文件
directory = "/home/ubuntu/Documents/llm/results_processed/"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        sample_token, scene_token = filename[:-4].split('_')
        print(f"Sample Token: {sample_token}, Scene Token: {scene_token}")
        if scene_token in folders_in_path2:
            # 计算并返回min_pet
            min_pet = evaluate_scene(sample_token, scene_token)

            # 将结果添加到DataFrame中
            new_row = pd.DataFrame({'scene_token': [scene_token], 'min_pet': [min_pet]})
            baseline_agent_driver_df = pd.concat([baseline_agent_driver_df, new_row], ignore_index=True)

# 计算每个scene_token的最小PET值
baseline_agent_driver_df_new = baseline_agent_driver_df.groupby('scene_token')['min_pet'].mean().reset_index()

# 读取场景信息
all_scenes_df = pd.read_csv('../scenes_info/all_scenes.csv')

# 合并场景信息
df = pd.merge(baseline_agent_driver_df_new, all_scenes_df, on='scene_token', how='left')

# 计算每种组合的最小PET值平均值
combinations_pet = df.groupby(['rain', 'intersection', 'night'])['min_pet'].min().reset_index()

# 绘制图像
min_pet = combinations_pet['min_pet'].min()
max_pet = combinations_pet['min_pet'].max()
plot_pet_matrix(combinations_pet, 'baseline_agent_driver', min_pet, max_pet, 'metrics_smoothness_Agent_Driver.png')



