import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion

# 初始化NuScenes对象
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ubuntu/Documents/Nuscenes', verbose=True)

# 定义文件路径
scenes_data_folder = "/home/ubuntu/Documents/VLM_MPC/scenes_data/"
results_processed_folder = "/home/ubuntu/Documents/llm/results_processed/"
scene_token = "5af9c7f124d84e7e9ac729fafa40ea01"

# 读取轨迹和其他车辆数据
vehs_trj_file_path = os.path.join(scenes_data_folder, scene_token, "vehs_trj.csv")
vehs_trj_df = pd.read_csv(vehs_trj_file_path)

# 绘图准备
fig, ax = plt.subplots(figsize=(10, 10))

# 循环遍历每个vehicle_id
for index, row in vehs_trj_df.iterrows():
    vehicle_id = row['vehicle_id']
    timestamp = row['timestamp']
    closest_sample = nusc.get('sample', nusc.get_closest_sample_token(scene_token, timestamp))
    annotations = nusc.get('sample_annotation', closest_sample['anns'])

    for annotation_token in annotations:
        annotation = nusc.get('sample_annotation', annotation_token)
        if annotation['instance_token'] == vehicle_id:
            # 获取 bounding box
            box = nusc.get_box(annotation_token)

            # 绘制 bounding box
            corners = box.corners()
            ax.plot(corners[0, :], corners[1, :], '-')

            # 如果需要渲染box的详细信息
            box.render(ax)

# 设置图形属性
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Vehicle Trajectories and Bounding Boxes')
ax.grid(True)
ax.axis('equal')
plt.show()
