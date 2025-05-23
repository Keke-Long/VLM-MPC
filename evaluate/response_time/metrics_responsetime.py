import os
import matplotlib.pyplot as plt
import numpy as np


base_path = '../../scenes_data/'
scene_folders = sorted(os.listdir(base_path))

# 用于存储所有response_time的列表
all_response_times = []

for scene_token_name in scene_folders:
    time_file = f"{base_path}{scene_token_name}/p_VLM_MPC1/llava/responsetime.txt"

    # 检查文件是否存在
    if os.path.exists(time_file):
        with open(time_file, "r") as file:
            times = file.readlines()
            # 将读取到的时间转换为浮点数并添加到列表中
            all_response_times.extend([float(time.strip()) for time in times])

# all_response_times = [time for time in all_response_times if time < 5.5]


# 绘制直方图
plt.figure(figsize=(3.5, 2.5))
plt.hist(all_response_times, bins=30, edgecolor='black', alpha=0.7, density=True)
plt.xlabel('Response Time (seconds)')
plt.ylabel('Frequency')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('responsetime p_VLM_MPC1 llava.png', dpi=300)
#plt.show()

