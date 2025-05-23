'''
给定token name，输出descripton
'''

from nuscenes.nuscenes import NuScenes

# 初始化NuScenes数据集
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

# 给定场景 token
scene_token = '0ac05652a4c44374998be876ba5cd6fd'

# 获取场景信息
scene = nusc.get('scene', scene_token)

# 获取描述信息
description = scene['description']

# 输出描述信息
print(f"Description for scene {scene_token}: {description}")
