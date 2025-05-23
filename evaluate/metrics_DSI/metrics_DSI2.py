'''
针对所有场景，根据场景特征分类后画热力图,
'''

import os
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt



method = 'realworld'
method = 'b_LLM_MPC/gpt-3.5-turbo'



def evaluate_DSI(scene_token_name, method, M=1, K=1, k=2):
    if method == 'realworld':
        file_path = f"../../scenes_data/{scene_token_name}/vehs_trj.csv"
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return np.nan
        data = pd.read_csv(file_path)
        data['delta_x'] = data.groupby('vehicle_id')['x'].diff()
        data['delta_y'] = data.groupby('vehicle_id')['y'].diff()
        data['delta_t'] = data.groupby('vehicle_id')['timestamp'].diff()
        data['velocity'] = np.sqrt(data['delta_x'] ** 2 + data['delta_y'] ** 2) / data['delta_t']

        results = pd.DataFrame()
        for time, group in data.groupby('timestamp'):
            vehicles_data = group[['vehicle_id', 'x', 'y', 'velocity']].dropna()
            total_dsi = 0
            for (id1, x1, y1, v1), (id2, x2, y2, v2) in combinations(vehicles_data.values, 2):
                if 'ego_vehicle' in [id1, id2]:
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if distance > 0:
                        potential_field = K * M * M / distance ** k
                        kinetic_field = K * M * M * np.abs(v1 - v2) / distance ** k
                        total_safety_index = potential_field + kinetic_field
                        total_dsi += total_safety_index

            results = pd.concat([results, pd.DataFrame({'total_dsi': [total_dsi]})], ignore_index=True)

    else:
        file_path = f"../../scenes_data/{scene_token_name}/{method}/Trj_result.csv"
        print('file_path',file_path)
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return np.nan
        data = pd.read_csv(file_path)
        data['delta_x'] = data.groupby('vehicle_id')['x_new'].diff()
        data['delta_y'] = data.groupby('vehicle_id')['y_new'].diff()
        data['delta_t'] = data.groupby('vehicle_id')['timestamp'].diff()
        data['velocity'] = np.sqrt(data['delta_x'] ** 2 + data['delta_y'] ** 2) / data['delta_t']

        results = pd.DataFrame()
        for time, group in data.groupby('timestamp'):
            vehicles_data = group[['vehicle_id', 'x_new', 'y_new', 'velocity']].dropna()
            total_dsi = 0
            for (id1, x1, y1, v1), (id2, x2, y2, v2) in combinations(vehicles_data.values, 2):
                if 'ego_vehicle' in [id1, id2]:
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if distance > 0:
                        potential_field = K * M * M / distance ** k
                        kinetic_field = K * M * M * np.abs(v1 - v2) / distance ** k
                        total_safety_index = potential_field + kinetic_field
                        total_dsi += total_safety_index

            results = pd.concat([results, pd.DataFrame({'total_dsi': [total_dsi]})], ignore_index=True)

    return results['total_dsi'].mean() if not results.empty else np.nan


# 函数：生成热力图并保存结果
def plot_dsi_heatmap(combinations_dsi, save_path):
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    for idx, rain_value in enumerate([0, 1]):
        ax = axes[idx]
        subset = combinations_dsi[combinations_dsi['rain'] == rain_value]
        dsi_matrix = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])

        for _, row in subset.iterrows():
            dsi_matrix.at[row['night'], row['intersection']] = row['average_dsi']

        cax = ax.imshow(dsi_matrix, cmap='viridis', vmin=dsi_matrix.min().min(), vmax=dsi_matrix.max().max())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
        ax.set_xlabel('Intersection')
        ax.set_ylabel('Night')
        ax.set_title(f'Rain={rain_value}')

        for (i, j), val in np.ndenumerate(dsi_matrix.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')

    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    data_dir = '../../scenes_data/'
    scenes_info_dir = '../../scenes_info/'
    all_scenes_df = pd.read_csv(f'{scenes_info_dir}/all_scenes.csv')
    scene_folders = set(os.listdir(data_dir))

    filtered_scenes_df = all_scenes_df[all_scenes_df['scene_token'].isin(scene_folders)]
    filtered_scenes_df['average_dsi'] = filtered_scenes_df['scene_token'].apply(lambda x: evaluate_DSI(x, method))

    combinations_dsi = filtered_scenes_df.groupby(['rain', 'intersection', 'night'])['average_dsi'].mean().reset_index()
    plot_dsi_heatmap(combinations_dsi, 'metrics_dsi.png')