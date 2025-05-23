'''
获取所有车轨迹
'''

import csv
import os
from nuscenes.nuscenes import NuScenes
import numpy as np


# 定义一个函数，将时间戳近似到最近的0.5秒
def round_to_nearest_half_second(timestamp):
    return np.round(timestamp * 2) / 2


def get_trj(scene_token, nusc):
    # # 加载nuScenes数据集
    # dataroot = '/home/ubuntu/Documents/Nuscenes'
    # nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


    # 获取指定场景
    scene = nusc.get('scene', scene_token)
    scene_folder = os.path.abspath(os.path.join('../scenes_data', scene_token))
    os.makedirs(scene_folder, exist_ok=True)  # 创建以场景token命名的文件夹


    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    trajectories = {}
    ego_trajectory = []


    while sample['next'] != '':
        timestamp = sample['timestamp'] / 1e6

        # 获取本车位置
        ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
        ego_translation = ego_pose['translation']
        ego_trajectory.append({
            'timestamp': timestamp,
            't': round_to_nearest_half_second(timestamp),
            'x': ego_translation[0],
            'y': ego_translation[1]
        })

        # 获取其他车辆位置
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            category = ann['category_name']
            if 'vehicle' in category:
                if instance_token not in trajectories:
                    trajectories[instance_token] = []
                translation = ann['translation']
                trajectories[instance_token].append({
                    'timestamp': timestamp,
                    't': round_to_nearest_half_second(timestamp),
                    'x': translation[0],
                    'y': translation[1]
                })
        sample = nusc.get('sample', sample['next'])

    # 保存轨迹数据到CSV文件
    csv_file = os.path.abspath(os.path.join(scene_folder, "vehs_trj.csv"))
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['vehicle_id', 'timestamp', 't', 'x', 'y'])
        # 写入本车轨迹
        for point in ego_trajectory:
            writer.writerow(['ego_vehicle', point['timestamp'], point['t'], point['x'], point['y']])
        # 写入其他车辆轨迹
        for instance, traj in trajectories.items():
            for point in traj:
                writer.writerow([instance, point['timestamp'], point['t'], point['x'], point['y']])



# if __name__ == "__main__":
#     scene_token_name = '02f1e5e2fc544798aad223f5ae5e8440'
#     get_trj(scene_token_name)
