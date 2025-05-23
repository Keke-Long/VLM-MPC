'''
处理一个sample_token的结果，转化坐标系到平面坐标，画图+保存
画图：包括这个sample_token里planned的轨迹，本车真实轨迹，和周围车轨迹
保存转化了坐标后的轨迹
基于的代码是 细粒度比较12.py
'''

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免图形界面相关的错误
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# 定义NuScenes数据集的路径
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


scenes_data_folder = "/home/ubuntu/Documents/VLM_MPC/scenes_data/"
results_processed_folder = "/home/ubuntu/Documents/llm/results_processed/"


def main(sample_token):
    # 文件路径
    json_file_path = f"/home/ubuntu/Documents/llm/results/{sample_token}.json"

    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    process_trajectory(data, sample_token, scenes_data_folder, results_processed_folder)

def process_trajectory(data, sample_token, scenes_data_folder, results_processed_folder):
    planned_trajectory = data['planning_target'].replace("Planned Trajectory:\n", "").strip("[]")
    planned_points = [tuple(map(float, point.strip("()").split(","))) for point in planned_trajectory.split("), (")]

    # 获取与sample token对应的场景信息
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    sample_timestamp = sample['timestamp'] / 1e6
    sample_timestamp = round(sample_timestamp * 2) / 2

    # 转换规划的轨迹点到全球坐标系
    ego_pose_token = sample['data']['CAM_FRONT']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    transform_matrix = get_transformation_matrix(ego_pose)
    planned_global_points = [transform_point(point, transform_matrix) for point in planned_points]

    planned_df = pd.DataFrame(planned_global_points, columns=['x', 'y'])
    planned_df['t'] = np.linspace(sample_timestamp + 0.5, sample_timestamp + 3, 6)
    planned_df.to_csv(f"{results_processed_folder}/{sample_token}_{scene_token}.csv", index=False)

    plot_trajectory(planned_global_points, sample_token, scenes_data_folder, scene_token, sample_timestamp, results_processed_folder)

def get_transformation_matrix(ego_pose):
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    transform_matrix = np.dot(np.eye(4), ego_rotation.transformation_matrix)
    transform_matrix[:3, 3] = ego_translation
    return transform_matrix

def transform_point(point, matrix):
    homogeneous_point = np.array([point[1], point[0], 0, 1])
    transformed_point = np.dot(matrix, homogeneous_point)
    return transformed_point[:2]

def plot_trajectory(planned_global_points, sample_token, scenes_data_folder, scene_token, sample_timestamp, results_processed_folder):
    vehs_trj_file_path = os.path.join(scenes_data_folder, scene_token, "vehs_trj.csv")
    try:
        vehs_trj_df = pd.read_csv(vehs_trj_file_path)
    except FileNotFoundError:
        print(f"未找到文件: {vehs_trj_file_path}")
        return

    filtered_df = vehs_trj_df[(vehs_trj_df['t'] >= sample_timestamp) & (vehs_trj_df['t'] <= sample_timestamp + 3)]
    fig, ax = plt.subplots(figsize=(10, 8))
    planned_x = [point[0] for point in planned_global_points]
    planned_y = [point[1] for point in planned_global_points]

    plt.plot(planned_x, planned_y, label="Planned Trajectory", color="blue", marker='.')

    for vehicle in filtered_df['vehicle_id'].unique():
        vehicle_traj = filtered_df[filtered_df['vehicle_id'] == vehicle]
        if vehicle == 'ego_vehicle':
            plt.scatter(vehicle_traj['x'], vehicle_traj['y'], color='black', label=f"Ego Vehicle")
        else:
            plt.plot(vehicle_traj['x'], vehicle_traj['y'], color='gray', marker='.', label=f"Surrounding Veh")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Comparison of Planned Trajectory and Other Vehicles")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_processed_folder}/{sample_token}_planned_trjs.png")


if __name__ == "__main__":
    directory_path = '/home/ubuntu/Documents/llm/results/'
    files = os.listdir(directory_path)
    json_files = [file.rstrip('.json') for file in files if file.endswith('.json') and not file.endswith('_time.json')]
    for sample_token in json_files:
        print(sample_token)
        main(sample_token)
