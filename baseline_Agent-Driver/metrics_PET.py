'''
对一个特定scene计算PET
'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def calculate_speeds(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    speeds = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2) / 0.5
    return speeds


def find_front_vehicle(ego_data, other_vehicles):
    front_vehicles = []
    for index, row in ego_data.iterrows():
        distances = other_vehicles.apply(lambda r: np.sqrt((float(r['x']) - float(row['x']))**2 + (float(r['y']) - float(row['y']))**2), axis=1)
        min_distance_idx = distances.idxmin()
        front_vehicles.append(other_vehicles.loc[min_distance_idx])
    return pd.DataFrame(front_vehicles)


def calculate_pet(ego_data, other_vehicles):
    # 确保数据类型正确
    ego_data[['x', 'y', 't']] = ego_data[['x', 'y', 't']].astype(float)
    other_vehicles[['x', 'y', 't']] = other_vehicles[['x', 'y', 't']].astype(float)

    # 计算后车速度
    ego_speeds = calculate_speeds(ego_data['x'].tolist(), ego_data['y'].tolist())
    ego_data['speed'] = ego_speeds

    # 将后车数据与前车数据根据时间戳进行合并
    merged_data = pd.merge(ego_data, other_vehicles, on='t', suffixes=('_ego', '_front'))

    # 计算每个时间点的距离
    distances = np.sqrt((merged_data['x_ego'] - merged_data['x_front'])**2 + (merged_data['y_ego'] - merged_data['y_front'])**2)

    # 计算PET，避免除以零，使用小的正数epsilon
    epsilon = 1e-6
    pets = distances / (merged_data['speed'] + epsilon)
    pets = pets[np.isfinite(pets)]  # 过滤掉无穷大和NaN值

    # 返回最小的PET值
    return np.min(pets) if len(pets) > 0 else None



# Load data
planned_trj_path = "/home/ubuntu/Documents/llm/results_processed/515ffe0f141445ed8e0de6e674b64060_5af9c7f124d84e7e9ac729fafa40ea01.csv"
vehs_trj_file_path = "/home/ubuntu/Documents/VLM_MPC/scenes_data/5af9c7f124d84e7e9ac729fafa40ea01/vehs_trj2.csv"

planned_trj = pd.read_csv(planned_trj_path)
vehs_trj = pd.read_csv(vehs_trj_file_path)

# Assuming the planned trajectory represents the ego vehicle
ego_data = planned_trj

# Filter other vehicles and avoid direct modifications
other_vehicles = vehs_trj[vehs_trj['vehicle_id'] != 'ego_vehicle'].copy()
#other_vehicles = vehs_trj[(vehs_trj['vehicle_id'] != 'ego_vehicle') & (vehs_trj['front_vehicle'] == 1)].copy()

# Calculate PET
min_pet = calculate_pet(ego_data, other_vehicles)
print(f"The minimum PET for the scene is: {min_pet}")
