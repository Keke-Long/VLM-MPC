'''
读取标定的参数 执行MPC保存结果
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxpy import *
from MPC_fun import mpc_fun
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

# 加载nuScenes数据集
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

tau = 0.5  # 引擎滞后时间常数
dt = 0.5  # 时间步长


base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
for scene_token_name in ['e5a3df5fe95149b5b974af1d14277ea7']:
    print(f'process {scene_token_name}')

    mpc_result_path = f"../scenes_data/{scene_token_name}/MPC_following_result.png"
    parameter_path = f'../scenes_data/{scene_token_name}/best_parameter.csv'

    df_params = pd.read_csv(parameter_path, header=None)
    (N, Q, R, Q_h, desired_speed, desired_headway) = tuple(df_params.iloc[-1].values)
    N = int(N)
    print('parameters', N, Q, R, Q_h, desired_speed, desired_headway)

    df = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")

    # 提取ego_vehicle的轨迹
    ego_vehicle_id = 'ego_vehicle'
    ego_df = df[df['vehicle_id'] == ego_vehicle_id].copy()

    # 获取场景信息
    scene = nusc.get('scene', scene_token_name)

    # 获取地图信息
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)

    # 获取包含ego vehicle位置的车道
    closest_lane = nusc_map.get_closest_lane(ego_df.iloc[0]['x'], ego_df.iloc[1]['y'], radius=5)
    if closest_lane:
        lane_record = nusc_map.arcline_path_3.get(closest_lane)
        poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
        lane_centerline = np.array(poses)
    else:
        print("No lane found for the given ego vehicle position.")
        lane_centerline = np.array([])

    # 计算原始轨迹的初始方向
    dx_ego = ego_df.iloc[-1]['x'] - ego_df.iloc[0]['x']
    dy_ego = ego_df.iloc[-1]['y'] - ego_df.iloc[0]['y']
    initial_direction_ego = np.arctan2(dy_ego, dx_ego)
    initial_direction_lane = initial_direction_ego  # 用原始轨迹拟合的方向

    # 检查方向是否需要转180度
    if np.abs(initial_direction_ego - initial_direction_lane) > np.pi / 2:
        initial_direction_lane += np.pi  # 转180度

    # 初始化轨迹
    x_new = [ego_df.iloc[0]['x']]
    y_new = [ego_df.iloc[0]['y']]
    vx = np.sqrt((ego_df.iloc[1]['x'] - ego_df.iloc[0]['x']) ** 2 + (ego_df.iloc[1]['y'] - ego_df.iloc[0]['y']) ** 2) / dt

    stop_line_info = ego_df[['timestamp', 't', 'stop_line_token', 'stop_line_need_stop']].dropna()
    stop_line_info = stop_line_info[stop_line_info['stop_line_need_stop'] == 1]

    dx = np.cos(initial_direction_lane)
    dy = np.sin(initial_direction_lane)

    for i in range(1, len(ego_df)):
        # 获取当前时刻的前车信息
        current_timestamp = ego_df.iloc[i]['timestamp']
        front_vehicle = df[(df['timestamp'] >= current_timestamp) & (df['front_vehicle'] == 1)]
        lead_info = None
        if len(front_vehicle) > 1:
            lead_x = front_vehicle.iloc[0]['x']
            lead_y = front_vehicle.iloc[0]['y']
            lead_v = np.sqrt((front_vehicle.iloc[1]['x'] - lead_x) ** 2 + (front_vehicle.iloc[1]['y'] - lead_y) ** 2) / dt
            d0 = np.sqrt((lead_x - ego_df.iloc[i]['x']) ** 2 + (lead_y - ego_df.iloc[i]['y']) ** 2)
            lead_info = (d0, lead_v)

        # 检查是否进入stop_line且需要停车
        stop_line_row = stop_line_info[stop_line_info['timestamp'] == current_timestamp]

        if (not stop_line_row.empty) and stop_line_row.iloc[0]['stop_line_need_stop']==1:
            stop_line_token = stop_line_row.iloc[0]['stop_line_token']
            # 假设停止位置前4米处有一辆静止的前车
            lead_x = ego_df.iloc[i]['x'] + 4 * dx
            lead_y = ego_df.iloc[i]['y'] + 4 * dy
            lead_v = 0
            d0 = np.sqrt((lead_x - ego_df.iloc[i]['x']) ** 2 + (lead_y - ego_df.iloc[i]['y']) ** 2)
            lead_info = (d0, lead_v)

        # 调用 MPC 控制
        d_control, v_control, u_control = mpc_fun(vx, lead_info,
                                                  N, dt, Q, R, Q_h, tau,
                                                  desired_speed, desired_headway)

        # 更新速度和位置
        vx += u_control[0] * dt
        # 更新x_new和y_new
        d_new_value = d_control[1].item() if isinstance(d_control[1], np.ndarray) else d_control[1]
        x_new.append(x_new[-1] + dx * d_new_value)
        y_new.append(y_new[-1] + dy * d_new_value)

    # 添加新轨迹到原始数据
    ego_df.loc[:, 'x_new'] = pd.Series(x_new[:len(ego_df)], index=ego_df.index)
    ego_df.loc[:, 'y_new'] = pd.Series(y_new[:len(ego_df)], index=ego_df.index)

    # 合并新轨迹到原始数据
    df.loc[df['vehicle_id'] == ego_vehicle_id, 'x_new'] = ego_df['x_new'].values
    df.loc[df['vehicle_id'] == ego_vehicle_id, 'y_new'] = ego_df['y_new'].values

    # 保存更新后的CSV文件
    updated_csv_file = f"../scenes_data/{scene_token_name}/vehs_trj_MPC_result.csv"
    df.to_csv(updated_csv_file, index=False)

    # 绘制轨迹
    plt.figure(figsize=(5, 5))
    # plt.plot(ego_df['x'], ego_df['y'], label='Original Trajectory', color='blue', linewidth=1, marker='o',
    #          markerfacecolor='none', alpha=0.9)
    plt.plot(ego_df['x_new'], ego_df['y_new'], label='Generated Trajectory', color='red', linewidth=1, marker='^',
             markerfacecolor='none', alpha=0.9)

    # 绘制车道中线
    # if len(lane_centerline) > 0:
    #     plt.plot(lane_centerline[:, 0], lane_centerline[:, 1], label='Lane Centerline', color='green', linewidth=2,
    #              linestyle='--')

    # 添加图例和标签
    #plt.legend()
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Comparison of Original and Generated Trajectories')
    plt.axis('equal')
    plt.grid(False)
    plt.savefig(f"../scenes_data/{scene_token_name}/MPC_following_result.png")