'''
分析一致性
'''
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler

# 定义参数的名称和标签列
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']
label_columns = ['rain', 'intersection', 'parking_lot', 'night']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)
all_labels = pd.DataFrame(columns=label_columns)

# 加载描述数据
df_description = pd.read_csv('../../scenes_info/all_scenes.csv')

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

                # 调整 R 和 Q_h
                last_row['R'] = last_row['R'] / (last_row['Q']+0.1)
                last_row['Q_h'] = last_row['Q_h'] / (last_row['Q']+0.1)
                last_row['Q'] = 1  # 固定 Q 为 1

                all_params = pd.concat([all_params, last_row], ignore_index=True)

                # 读取对应的label_columns这些01变量
                labels = df_description[df_description['scene_token'] == scene_token_name][label_columns]
                if not labels.empty:
                    all_labels = pd.concat([all_labels, labels], ignore_index=True)

# 处理MPC参数数据
all_params = all_params.apply(pd.to_numeric, errors='coerce')
all_params = all_params.dropna()
scaler = StandardScaler()
scaled_params = scaler.fit_transform(all_params)

# 初始化结果字典
results = {label: {} for label in label_columns}

# 对每个标签进行分析
for label in label_columns:
    group_0 = all_params[all_labels[label] == 0]
    group_1 = all_params[all_labels[label] == 1]

    for param in parameter_names:
        t_stat, p_value = ttest_ind(group_0[param], group_1[param], equal_var=False)
        results[label][param] = p_value

# 转换结果为DataFrame
results_df = pd.DataFrame(results)

# 保存结果到CSV文件
results_df.to_csv('mpc_parameter_comparison.csv', index=False)

print("P-value analysis for MPC parameters based on different labels:")
for label in label_columns:
    print(f"\nLabel: {label}")
    for param in parameter_names:
        p_value = results_df.at[param, label]
        if p_value > 0.05:
            mean_value = all_params[param].mean()
            print(f"  MPC parameter: {param}, P-value: {p_value:.5f}, Conclusion: No significant difference, Mean: {mean_value:.5f}")
        else:
            mean_value_group_0 = group_0[param].mean()
            mean_value_group_1 = group_1[param].mean()
            print(f"  MPC parameter: {param}, P-value: {p_value:.5f}, Conclusion: Significant difference")
            print(f"    Mean (group 0): {mean_value_group_0:.5f}")
            print(f"    Mean (group 1): {mean_value_group_1:.5f}")
