import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_for_scenes(data_dir):
    mse_list = []
    outlier_scenes = []
    scene_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for scene_token_name in scene_folders:
        csv_path = os.path.join(data_dir, scene_token_name, 'vehs_trj_MPC_result.csv')
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            ego_data = data[data['vehicle_id'] == 'ego_vehicle']

            if not ego_data.empty:
                mse_total = np.mean((ego_data['x'] - ego_data['x_new']) ** 2 + (ego_data['y'] - ego_data['y_new']) ** 2)
                if mse_total <= 50:
                    outlier_scenes.append(scene_token_name)
                mse_list.append(mse_total)
            else:
                print(f"No ego_vehicle data in {csv_path}")
        else:
            print(f"CSV file does not exist: {csv_path}")

    return mse_list, outlier_scenes


def plot_mse_histogram(mse_list):
    plt.figure(figsize=(5, 4))
    plt.hist([mse for mse in mse_list], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('MSE')
    plt.ylabel('Number of Scenes')
    plt.title('Histogram of MSE for All Scenes (0-100)')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig('metrics_error.png')


# 主程序
data_dir = '../scenes_data1/'
mse_list, outlier_scenes = calculate_mse_for_scenes(data_dir)
plot_mse_histogram(mse_list)

if outlier_scenes:
    outlier_scenes.sort()  # 按顺序排序
    df_outliers = pd.DataFrame(outlier_scenes)
    df_outliers.to_csv('ok_night_scenes.csv', index=False, header=False)

    print("Scenes with MSE greater than threshold:")
    for scene in outlier_scenes:
        print(scene)
