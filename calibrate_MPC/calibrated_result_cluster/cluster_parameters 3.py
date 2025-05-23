'''
分组直方图
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义参数的名称和标签列
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']
label_columns = ['rain', 'intersection', 'parking_lot']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)
all_labels = pd.DataFrame(columns=label_columns)

# 加载描述数据
df_description = pd.read_csv('../../scenes_cluster/all_scenes.csv')

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

# 绘制直方图
def plot_histograms(param_data, label_data, label_name, parameter_names):
    fig, axs = plt.subplots(2, 6, figsize=(18, 10))
    fig.suptitle(f'Histograms of MPC Parameters by {label_name}', fontsize=16)

    for i, param in enumerate(parameter_names):
        group_0 = param_data[label_data == 0]
        group_1 = param_data[label_data == 1]

        axs[0, i].hist(group_0[param], bins=10, alpha=0.5, edgecolor='black', label=f'{label_name} = 0')
        axs[0, i].set_title(f'{param} ({label_name} = 0)')
        axs[0, i].set_xlabel('Value')
        axs[0, i].set_ylabel('Frequency')

        axs[1, i].hist(group_1[param], bins=10, alpha=0.5, edgecolor='black', label=f'{label_name} = 1')
        axs[1, i].set_title(f'{param} ({label_name} = 1)')
        axs[1, i].set_xlabel('Value')
        axs[1, i].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{label_name}_parameter_histograms.png')
    plt.show()

# 分别对比不同标签的直方图
for label in label_columns:
    plot_histograms(all_params, all_labels[label], label, parameter_names)
