'''
把所有存在的best_parameter.csv的最后一行的参数集合到一起，然后画出所有参数的直方图，
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


# 读取ok_night_scenes.csv文件中的场景名称
ok_night_scenes_path = '../ok_night_scenes.csv'
ok_night_scenes = pd.read_csv(ok_night_scenes_path, header=None).iloc[:, 0].tolist()


# 定义参数的名称
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)

# 遍历 ../../scenes_data/ 文件夹中的每个子文件夹
base_path = '../../scenes_data1/'
for scene_token_name in os.listdir(base_path):
    if scene_token_name in ok_night_scenes:

        scene_path = os.path.join(base_path, scene_token_name)
        if os.path.isdir(scene_path):
            best_param_file_path = os.path.join(scene_path, 'best_parameter.csv')
            if os.path.exists(best_param_file_path):
                # 读取 best_parameter.csv 文件
                df = pd.read_csv(best_param_file_path, header=None, names=parameter_names)
                if not df.empty:  # 检查文件是否为空
                    last_row = df.iloc[[-1]]  # 这里使用双括号来确保返回的是 DataFrame

                    # 调整 R 和 Q_h

                    last_row['R'] = last_row['R'] / (last_row['Q']+0.1)
                    last_row['Q_h'] = last_row['Q_h'] / (last_row['Q']+0.1)
                    last_row['Q'] = 1  # 固定 Q 为 1

                    all_params = pd.concat([all_params, last_row], ignore_index=True)

# 移除包含空值或非数值数据的行
all_params = all_params.dropna()
all_params = all_params.apply(pd.to_numeric, errors='coerce')
all_params = all_params.dropna()
print(all_params)

# 绘制所有参数的直方图
fig, axs = plt.subplots(2, 3, figsize=(8, 5))
fig.suptitle('Distribution of MPC Parameters')

for i, param in enumerate(parameter_names):
    row = i // 3
    col = i % 3
    axs[row, col].hist(all_params[param], bins=10, edgecolor='black')
    axs[row, col].set_title(param)
    axs[row, col].set_xlabel('Value')
    axs[row, col].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Distribution_MPC_Parameters.png')
plt.show()


# 打印每个参数的均值
for param in parameter_names:
    mean_value = all_params[param].mean()
    print(f"The mean of {param} is {mean_value}")