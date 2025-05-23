'''
把所有存在的best_parameter.csv的最后一行的参数集合到一起，然后画出所有参数的直方图，
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义参数的名称
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)

# 遍历 ../../scenes_data/ 文件夹中的每个子文件夹
base_path = '../scenes_data/'
for scene_token_name in os.listdir(base_path):
    best_param_file_path = f"../scenes_data/{scene_token_name}/llm1/extracted_parameters.csv"

    # 读取 best_parameter.csv 文件
    df = pd.read_csv(best_param_file_path, header=None)
    if not df.empty:  # 检查文件是否为空
        last_row = df.iloc[[-1]]  # 这里使用双括号来确保返回的是 DataFrame
        print(scene_token_name,last_row)
        last_row.columns = parameter_names  # 设置列名

        all_params = pd.concat([all_params, last_row], ignore_index=True)

# 移除包含空值或非数值数据的行
all_params = all_params.dropna()
all_params = all_params.apply(pd.to_numeric, errors='coerce')
all_params = all_params.dropna()

# 绘制所有参数的直方图
fig, axs = plt.subplots(2, 3, figsize=(8, 5))
fig.suptitle('Distribution of llm-MPC Parameters')

for i, param in enumerate(parameter_names):
    row = i // 3
    col = i % 3
    axs[row, col].hist(all_params[param], bins=10, edgecolor='black')
    axs[row, col].set_title(param)
    axs[row, col].set_xlabel('Value')
    axs[row, col].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Distribution_llmMPC_Parameters.png')
plt.show()
