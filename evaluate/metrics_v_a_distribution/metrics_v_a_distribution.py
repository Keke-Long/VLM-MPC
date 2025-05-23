import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(formatter={'float': lambda x: f"{x:.4f}"})


def calculate_speeds_and_accelerations(x, y, dt=0.5):
    # 计算速度
    dx = np.diff(x)
    dy = np.diff(y)
    speeds = np.sqrt(dx**2 + dy**2) / dt
    # 计算加速度
    accelerations = np.diff(speeds) / dt
    return speeds, accelerations


method = 'realworld'  # 示例方法名
#method = 'b_MPC'
#method = 'b_LLM_MPC'
#method = 'p_VLM_MPC1'

#for method in ['realworld', 'b_MPC', 'b_LLM_MPC', 'p_VLM_MPC1']:

# 初始化列表用于存储所有文件的速度和加速度数据
all_velocities = []
all_accelerations = []
extreme_acceleration_files = [] #加速度超出范围的文件名

# 文件夹路径
base_path = '../../scenes_data/'
scene_folders = sorted(os.listdir(base_path))


for scene_token_name in scene_folders:
    # print(f'Processing {scene_token_name}')
#for scene_token_name in ['e1e664292aa144bc8d0d5d6441df084c']:
    file_path = None
    coord_columns = []

    if method == 'realworld':
        file_path = f"{base_path}{scene_token_name}/vehs_trj.csv"
        coord_columns = ['x', 'y']

    elif method == 'b_MPC':
        file_path = f"{base_path}{scene_token_name}/b_MPC/MPC_result.csv"
        coord_columns = ['x_new', 'y_new']

    elif method == 'b_LLM_MPC':
        file_path = f"{base_path}{scene_token_name}/b_LLM_MPC/gpt-3.5-turbo/Trj_result.csv"
        coord_columns = ['x_new', 'y_new']

    elif method == 'p_VLM_MPC1':
        file_path = f"{base_path}{scene_token_name}/p_VLM_MPC2/gpt-4o/MPC_result.csv"
        coord_columns = ['x_new', 'y_new']

    # 检查文件是否存在并处理
    if file_path and os.path.exists(file_path) and os.path.exists(f"{base_path}{scene_token_name}/p_VLM_MPC2/gpt-4o/MPC_result.csv"):
        df = pd.read_csv(file_path)
        ego_vehicle = df[df['vehicle_id'] == 'ego_vehicle']

        # 将DataFrame转换为numpy数组
        data_array = ego_vehicle[coord_columns].to_numpy()
        x, y = data_array[:, 0], data_array[:, 1]

        velocities, accelerations = calculate_speeds_and_accelerations(x, y)
        all_velocities.extend(velocities)
        all_accelerations.extend(accelerations)
        # print(x)
        # print(y)
        # print(all_velocities)
        # print(all_accelerations)

        # 检查加速度是否超出范围
        if np.any((accelerations < -5) | (accelerations > 5)):
            extreme_acceleration_files.append(scene_token_name)



# 打印超出加速度范围的文件名
if extreme_acceleration_files:
    print("以下文件的加速度超过了的范围：")
    for file in extreme_acceleration_files:
        print(file)


# 打印速度和加速度的范围
all_velocities = np.array(all_velocities)
all_accelerations = np.array(all_accelerations)
max_v = all_velocities.max()
min_v = all_velocities.min()
max_a = all_accelerations.max()
min_a = all_accelerations.min()


# 画图
plt.figure(figsize=(6, 2.5))

plt.subplot(1, 2, 1)
plt.hist(all_velocities, bins=50, color='lightseagreen', alpha=0.7, density=True)
plt.xlabel("Speed (m/s)")
plt.ylabel("Frequency")
plt.xlim(0,25)
plt.title(f"min_v={min_v:.2f}, max_v={max_v:.2f}", fontsize=10, ha='center')

plt.subplot(1, 2, 2)
plt.hist(all_accelerations, bins=50, color='olivedrab', alpha=0.7, density=True)
plt.xlabel("Acceleration (m/s^2)")
plt.ylabel("Frequency")
#plt.xlim(-4,4)
plt.title(f"min_a={min_a:.2f}, max_a={max_a:.2f}", fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(f'v_a_distribution_{method}.png')

