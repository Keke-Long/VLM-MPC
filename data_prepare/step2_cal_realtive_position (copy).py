import os
import numpy as np
import pandas as pd

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])

def cal_relative_position(scene_token):
    # 加载轨迹数据
    input_csv_file = os.path.abspath(os.path.join('../scenes_data', scene_token, "vehs_trj.csv"))
    df = pd.read_csv(input_csv_file)

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

        # 提取当前时间点的所有车辆数据
        current_data = df[df['t'] == rounded_timestamp]
        for _, row in current_data.iterrows():
            if row['vehicle_id'] == 'ego_vehicle':
                df.loc[(df['vehicle_id'] == row['vehicle_id']) & (df['timestamp'] == row['timestamp']), ['delta_x', 'delta_y']] = [0, 0]
            else:
                translation = np.array([row['x'], row['y']])
                relative_translation = translation - ego_translation
                df.loc[(df['vehicle_id'] == row['vehicle_id']) & (df['timestamp'] == row['timestamp']), ['delta_x', 'delta_y']] = relative_translation

    # 保存相对位置数据到CSV文件
    output_csv_file = os.path.abspath(os.path.join(scene_folder, "vehs_trj2.csv"))
    df.to_csv(output_csv_file, index=False)

    print(f"Relative positions saved to {output_csv_file}")

if __name__ == "__main__":
    scene_token_name = 'e5a3df5fe95149b5b974af1d14277ea7'
    cal_relative_position(scene_token_name)
