import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 从metrics_smoothness.py中提取的平滑度计算方法
def evaluate_scene(scene_token_name, method, dt = 0.5):
    if method == 'realworld':
        file_path = f"../../scenes_data/{scene_token_name}/vehs_trj.csv"
        if not os.path.exists(file_path):
            return np.nan
        data = pd.read_csv(file_path)
        ego_data = data[data['vehicle_id'] == 'ego_vehicle']
        dx = np.diff(ego_data['x'])
        dy = np.diff(ego_data['y'])

    else:
        file_path = f"../../scenes_data/{scene_token_name}/{method}/{model}/Trj_result.csv"
        if not os.path.exists(file_path):
            return np.nan
        data = pd.read_csv(file_path)
        ego_data = data[data['vehicle_id'] == 'ego_vehicle']
        dx = np.diff(ego_data['x_new'])
        dy = np.diff(ego_data['y_new'])
    # 计算加速度并获取RMS值
    ddx = np.diff(dx) / dt
    ddy = np.diff(dy) / dt
    acceleration = np.sqrt(ddx ** 2 + ddy ** 2)
    rms_value = np.sqrt(np.mean(acceleration ** 2))
    return rms_value



# 定义计算方法
# method = 'realworld'
method = "b_LLM_MPC"
#model = "gpt-4o"
# model = "llama3"
model = "llama3_no_memory"
# model = "llama3_no_environment"



# 读取all_scenes.csv文件
all_scenes_df = pd.read_csv('../../scenes_info/all_scenes.csv')

# 获取../scenes_data/文件夹下的文件名集合
scenes_data_path = '../../scenes_data/'
set1 = set(os.listdir(scenes_data_path))

# 筛选出文件名在set1中的场景
filtered_scenes_df = all_scenes_df[all_scenes_df['scene_token'].isin(set1)]

# 计算每个场景的平滑度
filtered_scenes_df['smoothness'] = filtered_scenes_df['scene_token'].apply(lambda x: evaluate_scene(x, method))

# 计算每种组合的平滑度平均值
combinations_smoothness = filtered_scenes_df.groupby(['rain', 'intersection', 'night'])[
    'smoothness'].mean().reset_index()


# 定义颜色映射，其中1.5对应白色
norm = mcolors.TwoSlopeNorm(vmin=0.3, vcenter=0.6, vmax=3)

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# 分别绘制rain=0和rain=1的情况
for idx, rain_value in enumerate([0, 1]):
    ax = axes[idx]
    subset = combinations_smoothness[combinations_smoothness['rain'] == rain_value]
    # 创建一个空白的2x2矩阵
    smoothness_matrix = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])
    # 填充矩阵
    for _, row in subset.iterrows():
        smoothness_matrix.at[row['night'], row['intersection']] = row['smoothness']
    # 绘制热力图
    cax = ax.imshow(smoothness_matrix, cmap='summer', norm=norm)
    # 设置刻度和标签
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Intersection')
    ax.set_ylabel('Night')
    ax.set_title(f'rain={rain_value}')
    # 在每个格子上显示样本数量
    for (i, j), val in np.ndenumerate(smoothness_matrix.values):
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
# 添加颜色条
# cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)
# cbar.ax.tick_params(labelsize=14)  # 设置颜色条刻度标签的字体大小
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'metrics_smoothness_{method}_{model}.png')
