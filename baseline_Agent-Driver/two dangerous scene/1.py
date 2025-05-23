import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap


# 定义NuScenes数据集和地图的路径
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)



#%%

sample_token = "9f3355442ce347be804a9880d086ca8f"
scene_token = "5af9c7f124d84e7e9ac729fafa40ea01"
map_name = 'boston-seaport'
# Front camera image file: samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151423512404.jpg
# Full path to the front camera image file: /home/ubuntu/Documents/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151423512404.jpg


# sample_token = "515ffe0f141445ed8e0de6e674b64060"
# scene_token = "5af9c7f124d84e7e9ac729fafa40ea01"
# map_name = 'boston-seaport'
#
# sample_token = "41664d45adc8443cb32b2020b37dbbf1"
# scene_token = "bd338b912ce9434995b29b6dac9fbf1d"
# map_name = 'singapore-hollandvillage'
# Front camera image file: samples/CAM_FRONT/n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799554662460.jpg
# Full path to the front camera image file: /home/ubuntu/Documents/Nuscenes/samples/CAM_FRONT/n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799554662460.jpg


# 获取场景对象
scene = nusc.get('scene', scene_token)

# 打印场景的描述
print("Scene Description:", scene['description'])
# 打印相关信息，可能有助于确定地图
print("Log Token:", scene['log_token'])
# 获取和打印log信息
log = nusc.get('log', scene['log_token'])
print("Location:", log['location'])



# baseline Agent Driver Planned Trajectory
file_path = f"/home/ubuntu/Documents/llm/results_processed/{sample_token}_{scene_token}.csv"
planned_df = pd.read_csv(file_path)
t_min = planned_df['t'].min()

# Other vehicle Trajectory
vehs_trj_file_path = f"/home/ubuntu/Documents/VLM_MPC/scenes_data/{scene_token}/vehs_trj.csv"
vehs_trj_df = pd.read_csv(vehs_trj_file_path)
vehs_trj_df = vehs_trj_df[(vehs_trj_df['t'] >= t_min - 0.5) & (vehs_trj_df['t'] < t_min + 3)]
print(vehs_trj_df)

# Proposed VLM-MPC Trajectory
proposed_trj_file_path = f"/home/ubuntu/Documents/VLM_MPC/scenes_data/{scene_token}/p_VLM_MPC2/gpt-4o/MPC_result.csv"
proposed_trj_file_path = f"/home/ubuntu/Documents/VLM_MPC/scenes_data/{scene_token}/MPC_calibration/vehs_trj_MPC_result.csv"
proposed_trj_df = pd.read_csv(proposed_trj_file_path)
proposed_trj_df = proposed_trj_df[(proposed_trj_df['t'] >= t_min - 0.5) & (proposed_trj_df['t'] < t_min + 3) & (proposed_trj_df['vehicle_id']=='ego_vehicle')]
print(proposed_trj_df)



# # 获取地图API对象
# nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
# # 渲染整个地图
# fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1,  alpha=0.05)
#
# # 绘制计划的轨迹
# plt.plot(planned_df['x'], planned_df['y'], linestyle='--', marker='o', markerfacecolor='none', color='forestgreen', label='Baseline Planned Trajectory')
#
# # 绘制其他车辆轨迹
# for vehicle_id, group in vehs_trj_df.groupby('vehicle_id'):
#     if vehicle_id == 'ego_vehicle':
#         plt.plot(group['x'], group['y'], linestyle='-', marker='s', markerfacecolor='none', linewidth=2, color='gray', label='Ego Vehicle Real Trajectory')
#     else:
#         plt.plot(group['x'], group['y'], linestyle='-', marker='.', color='dimgray', label=f'Vehicle {vehicle_id}')

# # 绘制计划的轨迹
# plt.plot(proposed_trj_df['x_new'], proposed_trj_df['y_new'], linestyle='-.', marker='o', markerfacecolor='none', color='blue', label='Proposed VLM-MPC Planned Trajectory')
#
# ax.set_xlabel("X Coordinate")
# ax.set_ylabel("Y Coordinate")
# ax.set_aspect('equal')
# plt.show()



# 获取与sample_token相关联的sample
sample = nusc.get('sample', sample_token)

# 检索前方摄像头的sample_data
cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

# 打印前方摄像头图像的文件路径
print("Front camera image file:", cam_front_data['filename'])

# 如果你想获取完整的文件路径，可以这样做：
full_path = os.path.join(dataroot, cam_front_data['filename'])
print("Full path to the front camera image file:", full_path)