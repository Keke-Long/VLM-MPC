import csv
import os
import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])


def cal_relative_position(scene_token_name, nusc):

    # 加载轨迹数据
    csv_path = f'../scenes_data1/{scene_token_name}/vehs_trj.csv'
    df = pd.read_csv(csv_path)

    # 添加 delta_x 和 delta_y 列
    df['delta_x'] = 0.0
    df['delta_y'] = 0.0

    # 提取自车轨迹
    ego_df = df[df['vehicle_id'] == 'ego_vehicle']

    # 计算相对位置
    for idx, ego_row in ego_df.iterrows():
        timestamp = ego_row['timestamp']
        rounded_timestamp = ego_row['t']
        ego_translation = np.array([ego_row['x'], ego_row['y']])

        # 查找最近的sample_data
        sample_data_tokens = nusc.field2token('sample_data', 'timestamp', int(timestamp * 1e6))
        if not sample_data_tokens:
            continue
        sample_data_token = sample_data_tokens[0]
        sample_data = nusc.get('sample_data', sample_data_token)
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_rotation = np.array(ego_pose['rotation'])
        ego_rotation_matrix = quaternion_to_rotation_matrix(ego_rotation)

        # 提取当前时间点的所有车辆数据
        current_data = df[df['t'] == rounded_timestamp]
        for _, row in current_data.iterrows():
            if row['vehicle_id'] == 'ego_vehicle':
                df.loc[(df['vehicle_id'] == row['vehicle_id']) & (df['timestamp'] == row['timestamp']), ['delta_x', 'delta_y']] = [0, 0]
            else:
                translation = np.array([row['x'], row['y']])
                relative_translation = translation - ego_translation
                relative_translation = np.dot(ego_rotation_matrix[:2, :2].T, relative_translation)
                df.loc[(df['vehicle_id'] == row['vehicle_id']) & (df['timestamp'] == row['timestamp']), ['delta_x', 'delta_y']] = relative_translation

    # 保存相对位置数据到CSV文件
    output_csv_file = csv_path.replace('.csv', '2.csv')
    df.to_csv(output_csv_file, index=False)

    print(f"Relative positions saved")



# if __name__ == "__main__":
#
#     # 加载nuScenes数据集
#     dataroot = '/home/ubuntu/Documents/Nuscenes'
#     nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
#
#     scene_token_name = 'e5a3df5fe95149b5b974af1d14277ea7'
#     cal_relative_position(scene_token_name, nusc)
