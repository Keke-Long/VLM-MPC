import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from controller_MPC.MPC_fun import mpc_fun

tau = 0.5  # 引擎滞后时间常数
dt = 0.5  # 时间步长

def mpc_for_one_scene(scene_token_name, params, nusc, nusc_map):
    # 从params中提取MPC参数
    N, Q, R, Q_h, desired_speed, desired_headway = params

    df = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")

    # 提取ego_vehicle的轨迹
    ego_vehicle_id = 'ego_vehicle'
    ego_df = df[df['vehicle_id'] == ego_vehicle_id]

    # 获取场景信息
    scene = nusc.get('scene', scene_token_name)

    # 获取某个时间点ego vehicle的位置
    ego_pose_token = nusc.get('sample', scene['first_sample_token'])['data']['LIDAR_TOP']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    ego_translation = ego_pose['translation']
    ego_position = ego_translation[:2]  # 忽略z轴，只取x和y

    # 获取包含ego vehicle位置的车道
    closest_lane = nusc_map.get_closest_lane(ego_position[0], ego_position[1], radius=5)
    if closest_lane:
        lane_record = nusc_map.arcline_path_3.get(closest_lane)
        poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
        lane_centerline = np.array(poses)
    else:
        print("No lane found for the given ego vehicle position.")
        lane_centerline = np.array([])

    #if len(lane_centerline) > 1:
    # # 计算车道中线的初始方向
    # dx_lane = lane_centerline[-1][0] - lane_centerline[0][0]
    # dy_lane = lane_centerline[-1][1] - lane_centerline[0][1]
    # initial_direction_lane = np.arctan2(dy_lane, dx_lane)

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

    stop_line_info = ego_df[['timestamp', 'stop_line_token', 'stop_line_need_stop']].dropna()
    stop_line_info = stop_line_info[stop_line_info['stop_line_need_stop'] == 1]

    for i in range(1, len(ego_df)):
        # 使用车道中线的方向进行更新
        dx = np.cos(initial_direction_lane)
        dy = np.sin(initial_direction_lane)

        # 获取当前时刻的前车信息
        current_timestamp = ego_df.iloc[i]['timestamp']
        front_vehicle = df[(df['timestamp'] == current_timestamp) & (df['front_vehicle'] == 1)]
        lead_info = None
        if not front_vehicle.empty:
            lead_x = front_vehicle.iloc[0]['x']
            lead_y = front_vehicle.iloc[0]['y']
            lead_v = np.sqrt((front_vehicle.iloc[0]['x'] - lead_x) ** 2 + (front_vehicle.iloc[0]['y'] - lead_y) ** 2) / dt
            lead_info = (lead_x, lead_y, lead_v)

        # 检查是否进入stop_line且需要停车
        stop_line_row = stop_line_info[stop_line_info['timestamp'] == current_timestamp]
        if not stop_line_row.empty:
            stop_line_token = stop_line_row.iloc[0]['stop_line_token']
            # 假设停止位置前4米处有一辆静止的前车
            lead_x = ego_df.iloc[i]['x'] - 4 * dx
            lead_y = ego_df.iloc[i]['y'] - 4 * dy
            lead_v = 0
            lead_info = (lead_x, lead_y, lead_v)

        # 调用 MPC 控制
        x_control, y_control, v_control, u_control, omega_control = mpc_fun(
            x_new[-1], y_new[-1], vx, initial_direction_lane, lead_info, N, dt, Q, R, Q_h, tau,
            desired_speed, desired_headway)

        # 转换NumPy数组到Python标量
        x_new_value = x_control[1].item() if isinstance(x_control[1], np.ndarray) else x_control[1]
        y_new_value = y_control[1].item() if isinstance(y_control[1], np.ndarray) else y_control[1]

        # 更新速度和位置
        vx += u_control[0] * dt
        x_new.append(x_new_value)
        y_new.append(y_new_value)
    # else:
    #     # 没有找到车道的情况，直接返回原始数据
    #     print("Skipping trajectory calculation due to missing lane information.")
    #     ego_df.loc[:, 'x_new'] = np.nan
    #     ego_df.loc[:, 'y_new'] = np.nan
    #     df.update(ego_df[['vehicle_id', 'timestamp', 'x_new', 'y_new']])
    #     return df

    # 添加新轨迹到原始数据
    ego_df = df[df['vehicle_id'] == 'ego_vehicle'].copy()
    ego_df.loc[:, 'x_new'] = pd.Series(x_new[:len(ego_df)], index=ego_df.index)
    ego_df.loc[:, 'y_new'] = pd.Series(y_new[:len(ego_df)], index=ego_df.index)

    # 合并新轨迹到原始数据
    df.loc[df['vehicle_id'] == ego_vehicle_id, 'x_new'] = np.nan
    df.loc[df['vehicle_id'] == ego_vehicle_id, 'y_new'] = np.nan
    df.update(ego_df[['vehicle_id', 'timestamp', 'x_new', 'y_new']])

    return df

