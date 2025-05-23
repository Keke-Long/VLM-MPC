import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from sklearn.metrics import mean_squared_error
from mpc_for_one_scene import mpc_for_one_scene


def evaluate_mpc_performance(df):
    # 计算预测轨迹与真实轨迹的均方误差
    ego_df = df[df['vehicle_id'] == 'ego_vehicle'].copy()
    mse = mean_squared_error(ego_df[['x', 'y']], ego_df[['x_new', 'y_new']])
    return mse


def calibrate_mpc_parameters(scene_token_name, nusc, num_iterations=50):
    best_params = None
    best_score = float('inf')

    # 参数搜索范围，使用 linspace 保证均匀分布
    N_range = np.linspace(6, 12, dtype=int)  # 整数
    # Q_range = np.linspace(0, 0.8, 20)  # 速度维持权重
    R_range = np.linspace(0.5, 2, 20)  # 控制量权重
    Q_h_range = np.linspace(0.5, 4, 20)  # Headway 维持权重
    desired_speed_range = np.linspace(0.1, 8, 10)
    desired_headway_range = np.linspace(1.5, 4, 20)

    scene = nusc.get('scene', scene_token_name)
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)

    # Random search over the parameter space
    for _ in tqdm(range(num_iterations), desc=f"Calibrating token {scene_token_name}"):
        N = np.random.choice(N_range)
        Q = 1
        R = round(np.random.choice(R_range),1)
        Q_h = round(np.random.choice(Q_h_range),1)
        desired_speed = round(np.random.choice(desired_speed_range),1)
        desired_headway = round(np.random.choice(desired_headway_range),1)
        params = (N, Q, R, Q_h, desired_speed, desired_headway)

        df = mpc_for_one_scene(scene_token_name, params, nusc, nusc_map)
        mse = evaluate_mpc_performance(df)

        if mse < best_score:
            best_score = mse
            best_params = params
            #print('Best parameters:', best_params)

            df_params = pd.DataFrame([best_params])
            with open(f'../scenes_data1/{scene_token_name}/best_parameter.csv', 'a') as f:
                df_params.to_csv(f, header=f.tell() == 0, index=False)

    return best_params



# 加载nuScenes数据集
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
for idx, scene_token_name in enumerate(scene_folders):
    # Check if the file exists
    best_parameter_file_path = f'{base_path}{scene_token_name}/best_parameter.csv'
    if os.path.exists(best_parameter_file_path):
        continue
    try:
        best_params = calibrate_mpc_parameters(scene_token_name, nusc)
        print(f"Best MPC Parameters={best_params} of scene{scene_token_name}")
    except Exception as e:
        print(f"Error processing scene {scene_token_name}: {e}")
        continue
