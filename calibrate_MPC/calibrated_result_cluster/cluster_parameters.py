import os
import numpy as np
import pandas as pd

# 读取数据
df_description = pd.read_csv('../../scenes_cluster/all_scenes.csv')
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']
label_columns = ['rain', 'intersection', 'parking_lot']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)
all_labels = pd.DataFrame(columns=label_columns)

# 遍历 ../../scenes_data/ 文件夹中的每个子文件夹
base_path = '../../scenes_data/'
for scene_token_name in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_token_name)
    if os.path.isdir(scene_path):
        best_param_file_path = os.path.join(scene_path, 'best_parameter.csv')
        if os.path.exists(best_param_file_path):
            # 读取 best_parameter.csv 文件，手动指定列名
            df = pd.read_csv(best_param_file_path, header=None, names=parameter_names)
            if not df.empty:  # 检查文件是否为空
                last_row = df.iloc[[-1]]  # 这里使用双括号来确保返回的是 DataFrame
                all_params = pd.concat([all_params, last_row], ignore_index=True)

                # 读取对应的label_columns这些01变量
                labels = df_description[df_description['scene_token'] == scene_token_name][label_columns]
                if not labels.empty:
                    all_labels = pd.concat([all_labels, labels], ignore_index=True)

# 合并参数和标签数据
data = pd.concat([all_params, all_labels], axis=1)

# 计算所有场景的平均参数值
global_means = data[parameter_names].mean()

# 根据标签计算分类平均值
rain_mean_desired_headway = data[data['rain'] == 1]['desired_headway'].mean()
no_rain_mean_desired_headway = data[data['rain'] == 0]['desired_headway'].mean()

intersection_mean_desired_speed = data[data['intersection'] == 1]['desired_speed'].mean()
no_intersection_mean_desired_speed = data[data['intersection'] == 0]['desired_speed'].mean()
intersection_mean_R = data[data['intersection'] == 1]['R'].mean()
no_intersection_mean_R = data[data['intersection'] == 0]['R'].mean()

parking_lot_mean_desired_speed = data[data['parking_lot'] == 1]['desired_speed'].mean()
no_parking_lot_mean_desired_speed = data[data['parking_lot'] == 0]['desired_speed'].mean()

# 更新参数值
for index, row in data.iterrows():
    if row['rain'] == 1:
        data.at[index, 'desired_headway'] = rain_mean_desired_headway
    else:
        data.at[index, 'desired_headway'] = no_rain_mean_desired_headway

    if row['intersection'] == 1:
        data.at[index, 'desired_speed'] = intersection_mean_desired_speed
        data.at[index, 'R'] = intersection_mean_R
    else:
        data.at[index, 'desired_speed'] = no_intersection_mean_desired_speed
        data.at[index, 'R'] = no_intersection_mean_R

    if row['parking_lot'] == 1:
        data.at[index, 'desired_speed'] = parking_lot_mean_desired_speed
    else:
        data.at[index, 'desired_speed'] = no_parking_lot_mean_desired_speed

# 输出结果
print("Updated parameters based on labels:")
print(data[parameter_names].mean())
