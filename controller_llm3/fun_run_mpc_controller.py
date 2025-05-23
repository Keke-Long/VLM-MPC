import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from controller_MPC.MPC_fun import mpc_fun

tau = 0.5  # 引擎滞后时间常数
dt = 0.5  # 时间步长

def run_mpc_controller(scene_token_name, result_path):
    file_path = os.path.join(result_path, 'extracted_parameters.csv')
    parameters_df = pd.read_csv(file_path, header=None)
    N, Q, R, Q_h, desired_speed, desired_headway = parameters_df.iloc[0]
    N = int(N)

    df = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")
    ego_vehicle_id = 'ego_vehicle'
    ego_df = df[df['vehicle_id'] == ego_vehicle_id].copy()

    # 计算原始轨迹的初始方向
    dx_ego = ego_df.iloc[-1]['x'] - ego_df.iloc[0]['x']
    dy_ego = ego_df.iloc[-1]['y'] - ego_df.iloc[0]['y']
    initial_direction_ego = np.arctan2(dy_ego, dx_ego)
    initial_direction_lane = initial_direction_ego  # 用原始轨迹拟合的方向

    # 检查方向是否需要转180度
    if np.abs(initial_direction_ego - initial_direction_lane) > np.pi / 2:
        initial_direction_lane += np.pi  # 转180度

    # 从llm2读取前5秒的数据
    llm2_result_path = f'../scenes_data/{scene_token_name}/p_VLM_MPC1/gpt-4o/MPC_result.csv'
    llm2_df = pd.read_csv(llm2_result_path)
    initial_timestamp = ego_df['t'].min()
    five_seconds_later = initial_timestamp + 5

    # 复制前5秒的数据
    pre_5s_data = llm2_df[(llm2_df['vehicle_id'] == ego_vehicle_id) & (llm2_df['t'] <= five_seconds_later)]
    ego_df = pd.concat([pre_5s_data, ego_df[ego_df['t'] > five_seconds_later]])

    # 初始化轨迹
    x_new = ego_df['x_new'].tolist()[:len(pre_5s_data)]
    y_new = ego_df['y_new'].tolist()[:len(pre_5s_data)]
    print('ego_df', ego_df)
    vx = np.sqrt((ego_df.iloc[9]['x_new'] - ego_df.iloc[10]['x_new']) ** 2 + (ego_df.iloc[9]['y_new'] - ego_df.iloc[10]['y_new']) ** 2) / dt

    stop_line_info = ego_df[['timestamp', 't', 'stop_line_token', 'stop_line_need_stop']].dropna()
    stop_line_info = stop_line_info[stop_line_info['stop_line_need_stop'] == 1]

    dx = np.cos(initial_direction_lane)
    dy = np.sin(initial_direction_lane)

    for i in range(len(pre_5s_data), len(ego_df)):
        # 获取当前时刻的前车信息
        current_timestamp = ego_df.iloc[i]['timestamp']
        front_vehicle = df[(df['timestamp'] == current_timestamp) & (df['front_vehicle'] == 1)]
        lead_info = None
        if len(front_vehicle) > 1:
            lead_x = front_vehicle.iloc[0]['x']
            lead_y = front_vehicle.iloc[0]['y']
            lead_v = np.sqrt((front_vehicle.iloc[1]['x'] - lead_x) ** 2 + (front_vehicle.iloc[1]['y'] - lead_y) ** 2) / dt
            d0 = np.sqrt((lead_x - ego_df.iloc[i]['x']) ** 2 + (lead_y - ego_df.iloc[i]['y']) ** 2)
            lead_info = (d0, lead_v)

        # 检查是否进入stop_line且需要停车
        stop_line_row = stop_line_info[stop_line_info['timestamp'] == current_timestamp]

        if (not stop_line_row.empty) and stop_line_row.iloc[0]['stop_line_need_stop'] == 1:
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
    updated_csv_file = os.path.join(result_path, 'MPC_result.csv')
    df.to_csv(updated_csv_file, index=False)
    print('updated_csv_file', updated_csv_file)

    # 绘制轨迹
    plt.figure(figsize=(5, 5))
    # plt.plot(ego_df['x'], ego_df['y'], label='Original Trajectory', color='blue', linewidth=1, marker='o',
    #          markerfacecolor='none', alpha=0.9)
    plt.plot(ego_df['x_new'], ego_df['y_new'], label='Generated Trajectory', color='red', linewidth=1, marker='^',
             markerfacecolor='none', alpha=0.9)

    # 添加图例和标签
    plt.legend()
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Comparison of Original and Generated Trajectories')
    plt.grid(False)
    plt.axis('equal')
    plt.savefig(f"{result_path}/MPC_result.png")
    plt.close()
