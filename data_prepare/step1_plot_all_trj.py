'''
画所有车轨迹
'''

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_trj(scene_token_name):
    scene_folder = os.path.join('../scenes_data1', scene_token_name)
    csv_file_path = os.path.abspath(os.path.join(scene_folder, 'vehs_trj.csv'))

    # 检查CSV文件是否存在
    if os.path.exists(csv_file_path):
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 创建绘图
        plt.figure(figsize=(8, 8))
        colors = plt.cm.get_cmap('hsv', len(df['vehicle_id'].unique()))

        ego_vehicle_id = 'ego_vehicle'  # 假设本车ID为 'ego_vehicle'

        # 绘制本车轨迹
        ego_data = df[df['vehicle_id'] == ego_vehicle_id]
        plt.scatter(ego_data['x'], ego_data['y'], color='black', label=ego_vehicle_id)

        # 绘制其他车辆轨迹
        for vehicle_id, vehicle_data in df.groupby('vehicle_id'):
            if vehicle_id != ego_vehicle_id:
                color = colors(list(df['vehicle_id'].unique()).index(vehicle_id))
                plt.plot(vehicle_data['x'], vehicle_data['y'], label=vehicle_id, color=color)

        #plt.title(f"Vehicle Trajectories for Scene: {scene_token}")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        #plt.legend(loc='upper right', fontsize='small')

        # 保存图像到相同的场景文件夹中
        plot_file_path = os.path.abspath(os.path.join(scene_folder, f"vehs_trj.png"))
        plt.savefig(plot_file_path)
        plt.close()

        print(f"Trajectory plot saved to {plot_file_path}")
    else:
        print(f"CSV file does not exist: {csv_file_path}")



# if __name__ == "__main__":
#     scene_token_name = 'e5a3df5fe95149b5b974af1d14277ea7'
#     plot_all_trj(scene_token_name)