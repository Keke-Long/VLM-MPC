'''
统计有多少文件标了，多少没标
'''

import os
import numpy as np
import pandas as pd

# 定义文件夹路径
base_path = '../scenes_data/'

# 初始化计数器
total_folders = 0
folders_with_mpc_result = 0
folders_with_best_params = 0
folders_without_best_params = []

# 遍历 base_path 文件夹中的每个子文件夹
for scene_token_name in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_token_name)
    if os.path.isdir(scene_path):
        total_folders += 1
        mpc_result_path = os.path.join(scene_path, "MPC_following_result.png")
        parameter_path = os.path.join(scene_path, "best_parameter.csv")
        if os.path.exists(mpc_result_path):
            folders_with_mpc_result += 1
        if os.path.exists(parameter_path):
            folders_with_best_params += 1
        else:
            folders_without_best_params.append(scene_token_name)

# 打印结果
print(f"Total number of folders: {total_folders}")
print(f"Number of folders with MPC_following_result.png: {folders_with_mpc_result}")
print(f"Number of folders with best_parameter.csv: {folders_with_best_params}")

df = pd.read_csv('../scenes_cluster/boston_scene_info.csv')
filtered_df = df[(df['turn'] == 0) & (df['checked_turn'] == 0) & (df['length_more_than_10'] == 1)]
print('应该标定的数量', len(filtered_df))

# 打印缺少 best_parameter.csv 的文件夹
print("Folders without best_parameter.csv:")
for folder in folders_without_best_params:
    print(folder)

