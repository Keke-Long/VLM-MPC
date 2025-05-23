'''
读取轨迹数据，计算本车的原始轨迹与前车在前车存在的每个时间点的PET(Post Encroachment Time)，在原始csv中添加一列称作‘PET_origin’
如果这个样本中已经规划了新的轨迹，再用新轨迹计算前车在前车存在的每个时间点的PET，在原始csv中添加一列称作‘PET_MPC’
两种PET的分布，画成一个直方图

检查异常
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_speeds(x, y):
    speeds = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / 0.5  # Speeds calculated between consecutive points
    speeds_padded = np.append(speeds, np.nan)  # Pad with NaN for the last point
    return speeds_padded


def evaluate_scene(scene_token_name, method):
    file_path = f"../../scenes_data/{scene_token_name}/{method}/Trj_result.csv"
    data = pd.read_csv(file_path, low_memory=False)
    # Remove existing PET columns if they exist
    data = data.drop(columns=[col for col in data.columns if col.startswith('PET_')], errors='ignore')

    # Identify rows where 'vehicle_id' is 'ego_vehicle' and 'front_vehicle' is 1
    ego_data = data[data['vehicle_id'] == 'ego_vehicle']
    front_data = data[data['front_vehicle'] == 1]

    # Ensure both datasets are sorted by time to align them by timestamp
    ego_data = ego_data.sort_values(by='timestamp')
    front_data = front_data.sort_values(by='timestamp')

    # Calculate speeds using the sorted ego vehicle data
    ego_data['speed'] = calculate_speeds(ego_data['x'], ego_data['y'])
    ego_data['speed_new'] = calculate_speeds(ego_data['x_new'], ego_data['y_new'])

    # Merge ego and front vehicle data based on timestamp to align them on the same time point
    merged_data = pd.merge(ego_data, front_data[['timestamp', 'x', 'y']],
                                on='timestamp', how='inner', suffixes=('_ego', '_front'))

    # Calculate distances using the positions of ego and front vehicles
    merged_data['distance'] = np.sqrt(
        (merged_data['x_ego'] - merged_data['x_front'])**2 + (merged_data['y_ego'] - merged_data['y_front'])**2)
    merged_data['distance_new'] = np.sqrt(
        (merged_data['x_new'] - merged_data['x_front'])**2 + (merged_data['y_new'] - merged_data['y_front'])**2)

    # Calculate PET using the distances and speeds
    merged_data['PET_origin'] = merged_data['distance'] / merged_data['speed']
    merged_data['PET_MPC'] = merged_data['distance_new'] / merged_data['speed_new']

    # Replace any infinite or NaN values with NaN and remove any NaN entries
    merged_data['PET_origin'].replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data['PET_MPC'].replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data.dropna(subset=['PET_origin', 'PET_MPC'], inplace=True)

    # Save PET_origin and PET_MPC columns to the original csv file
    data = data.merge(merged_data[['timestamp', 'PET_origin', 'PET_MPC']], on='timestamp', how='left')
    data.to_csv(file_path, index=False)

    condition = merged_data['PET_MPC'].min() < 1
    return condition, merged_data


data_dir = '../../scenes_data/'
scene_folders = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])

condition_count = 0

# method = 'b_MPC'
# method = 'p_VLM_MPC1/llama3_no_memory'
method = 'b_LLM_MPC/llama3_no_memory'

for scene_token_name in scene_folders:
    # print('process', scene_token_name)
    if os.path.isfile(f"../../scenes_data/{scene_token_name}/{method}/Trj_result.csv"):
        condition, merged_data = evaluate_scene(scene_token_name, method)
        if condition:
            print(scene_token_name)
            condition_count += 1
            # Plotting both PET distributions on the same plot
            plt.figure(figsize=(5, 3))
            bins_range = np.linspace(0, 50, 151)  # 50 bins from 0 to 10 seconds

            plt.hist(merged_data['PET_origin'], bins=bins_range, color='gray', alpha=0.5, label='real-world')
            plt.hist(merged_data['PET_MPC'], bins=bins_range, color='blue', alpha=0.5, label=f"{method}")

            plt.title(f'Histogram of PET_origin and PET_MPC\n '
                      f'PET Range of real-world: {merged_data["PET_origin"].min():.2f}-{merged_data["PET_origin"].max():.2f}s \n'
                      f'PET Range of {method}: {merged_data["PET_MPC"].min():.2f}-{merged_data["PET_MPC"].max():.2f}s')
            plt.xlabel('PET Value (s)')
            plt.ylabel('Frequency')
            plt.xlim([0, 50])
            plt.legend()
            plt.tight_layout()
            plt.show()

print(f"Total scenes evaluated: {len(scene_folders)}")
print(f"Number of scenes where PET_origin.min() > PET_MPC.min() and PET_MPC.min() < 1: {condition_count}")
