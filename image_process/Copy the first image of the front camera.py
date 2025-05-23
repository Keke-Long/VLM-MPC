from nuscenes.nuscenes import NuScenes
import os
import shutil


# 指定场景token和文件夹路径
scene_token_name = '48ba943c3d19463a81281bf6a7078eac'
scene_token_name = 'e5a3df5fe95149b5b974af1d14277ea7'
scene_file = f"../scenes_data/{scene_token_name}/"

# 加载nuScenes数据集
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

# 创建目标文件夹如果它不存在
if not os.path.exists(scene_file):
    os.makedirs(scene_file)

# 获取场景信息
scene = nusc.get('scene', scene_token_name)
sample_token = scene['first_sample_token']

# 获取第一张前方摄像头图片,复制到目标文件夹
sample = nusc.get('sample', sample_token)
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
cam_filepath = nusc.get_sample_data_path(cam_token)

shutil.copy(cam_filepath, scene_file)

print(f"First front camera image has been copied to {scene_file}")
