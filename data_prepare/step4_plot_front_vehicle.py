'''
画图展示前车
    用黑色散点表示本车位置。
    用彩色散点表示其他车辆位置。
    用半透明的灰色线连接本车位置与前车位置。
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_front_vehicle(scene_token_name):

    csv_path = f'../scenes_data1/{scene_token_name}/vehs_trj2.csv'
    df = pd.read_csv(csv_path)

    # 确定本车ID（假设本车ID为第一个时间点的第一个车辆）
    ego_vehicle_id = df.loc[df['timestamp'] == df['timestamp'].min(), 'vehicle_id'].values[0]

    # 计算每个时间点的前车信息
    timestamps = df['timestamp'].unique()

    # 创建保存图片的文件夹
    output_folder = f'../scenes_data/{scene_token_name}/images_of_front_vehicle'
    os.makedirs(output_folder, exist_ok=True)

    # 可视化每个时间点的前车识别结果
    for timestamp in timestamps:
        plt.figure(figsize=(4, 4), dpi=50)

        # 绘制本车位置
        ego_position = df[(df['timestamp'] == timestamp) & (df['vehicle_id'] == ego_vehicle_id)]
        plt.scatter(ego_position['x'], ego_position['y'], color='black', label='Ego Vehicle', s=30)

        # 绘制其他车辆位置
        current_data = df[df['timestamp'] == timestamp]
        for _, row in current_data.iterrows():
            if row['vehicle_id'] != ego_vehicle_id:
                plt.scatter(row['x'], row['y'], label=f'Vehicle {row["vehicle_id"]}', alpha=0.7)

        # 绘制本车与前车的连线
        front_vehicles = current_data[current_data['front_vehicle'] == 1]
        if not ego_position.empty:
            ego_x = ego_position['x'].values[0]
            ego_y = ego_position['y'].values[0]
            for _, front_vehicle in front_vehicles.iterrows():
                plt.plot([ego_x, front_vehicle['x']], [ego_y, front_vehicle['y']], color='gray', alpha=0.5)

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.tight_layout()
        # 保存图片
        plt.savefig(os.path.join(output_folder, f"{timestamp}.png"), quality=10, optimize=True)
        plt.close()


# if __name__ == "__main__":
#     scene_token_name = '02f1e5e2fc544798aad223f5ae5e8440'
#     plot_front_vehicle(scene_token_name)
