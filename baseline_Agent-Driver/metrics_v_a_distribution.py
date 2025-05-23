import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件夹路径
results_folder = "/home/ubuntu/Documents/llm/results/"

# 初始化列表用于存储所有文件的速度和加速度数据
all_velocities = []
all_accelerations = []
extreme_acceleration_files = [] #加速度超出范围的文件名

# 获取文件夹中的所有文件名
files = [file for file in os.listdir(results_folder) if not file.endswith('time.json')]

# 遍历每个文件
for file_name in files:
    try:
        file_path = os.path.join(results_folder, file_name)

        # 读取JSON文件
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 提取“Planned Trajectory”
        trajectory = data.get("planning_target", "").replace("Planned Trajectory:\n", "").strip("[]")
        trajectory_points = [tuple(map(float, point.strip("()").split(","))) for point in trajectory.split("), (")]

        # 计算速度和加速度，假设时间步长为0.5秒
        velocities = []
        accelerations = []
        time_step = 0.5
        for i in range(1, len(trajectory_points)):
            x1, y1 = trajectory_points[i - 1]
            x2, y2 = trajectory_points[i]

            # 计算速度
            velocity = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / time_step
            velocities.append(velocity)

            if i > 1:
                # 计算加速度
                acceleration = (velocity - velocities[-2]) / time_step
                accelerations.append(acceleration)

                # 检查加速度是否超出范围 [-5, 5]
                if acceleration < -10 or acceleration > 6:
                    extreme_acceleration_files.append(file_name)

        # 将每个文件的速度和加速度添加到总列表中
        all_velocities.extend(velocities)
        all_accelerations.extend(accelerations)

    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        continue


# 打印超出加速度范围的文件名
if extreme_acceleration_files:
    print("以下文件的加速度超过了[-5, 5]的范围：")
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
plt.xlim(-25,10)
plt.title(f"min_a={min_a:.2f}, max_a={max_a:.2f}", fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('metrics_v_a_distribution.png')
