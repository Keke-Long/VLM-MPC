import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from scipy.spatial import KDTree

# 设置dataroot和初始化NuScenes
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
nusc_map = NuScenesMap(dataroot=dataroot, map_name='boston-seaport')

base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))

for idx, scene_token_name in enumerate(scene_folders):
    scene_path = os.path.join(base_path, scene_token_name)
    output_csv = os.path.join(scene_path, 'lane_centerlines.csv')
    output_image = os.path.join(scene_path, 'lane_centerlines.png')

    # 提取场景信息
    scene = nusc.get('scene', scene_token_name)
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    # 提取本车轨迹
    ego_vehicle_id = 'ego_vehicle'
    ego_traj = []

    while sample['next'] != '':
        ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
        ego_traj.append([ego_pose['translation'][0], ego_pose['translation'][1]])
        sample = nusc.get('sample', sample['next'])

    ego_traj = np.array(ego_traj)

    # 获取对应的道路中线
    lane_centerlines = []

    for ego_position in ego_traj:
        closest_lane = nusc_map.get_closest_lane(ego_position[0], ego_position[1], radius=5)
        if closest_lane:
            lane_record = nusc_map.arcline_path_3.get(closest_lane)
            poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
            for pose in poses:
                lane_centerlines.append(pose[:2])  # 只保存x和y列

    if len(lane_centerlines) > 0:
        lane_centerlines = np.array(lane_centerlines)

        # 构建KDTree用于快速最近邻搜索
        lane_centerline_tree = KDTree(lane_centerlines)

        # 找到每个轨迹点对应的最近的中线点
        nearest_lane_points = []
        for point in ego_traj:
            distance, index = lane_centerline_tree.query(point)
            nearest_lane_points.append(lane_centerlines[index])

        # 保存为CSV文件
        nearest_lane_points_df = pd.DataFrame(nearest_lane_points, columns=['x', 'y'])
        nearest_lane_points_df.to_csv(output_csv, index=False)

        # 画图展示并保存
        plt.figure(figsize=(10, 8))
        plt.plot(ego_traj[:, 0], ego_traj[:, 1], 'r', label='Ego Vehicle Trajectory')
        nearest_lane_points_df = nearest_lane_points_df.drop_duplicates()
        plt.scatter(nearest_lane_points_df['x'], nearest_lane_points_df['y'], s=1, c='b', label='Lane Centerlines')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Ego Vehicle Trajectory and Lane Centerlines for Scene {scene_token_name}')
        plt.savefig(output_image)
        plt.close()

        print(f'Processed scene {scene_token_name} ({idx+1}/{len(scene_folders)})')
    else:
        print(f'No lane centerlines found for scene {scene_token_name} ({idx+1}/{len(scene_folders)})')
