'''
挨个显示图片，检查效果
'''

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


base_path = '../scenes_data1/'

ok_night_scenes_path = 'ok_night_scenes.csv'
ok_night_scenes = pd.read_csv(ok_night_scenes_path, header=None).iloc[:, 0].tolist()


# 初始化计数器
total_folders = 0
folders_with_mpc_result = 0
folders_with_best_params = 0


base_path = '../scenes_data1/'
scene_folders = sorted(os.listdir(base_path))
for scene_token_name in scene_folders:
    if scene_token_name in ok_night_scenes:
        print(f'process {scene_token_name}')

        scene_path = os.path.join(base_path, scene_token_name)
        if os.path.isdir(scene_path):
            total_folders += 1
            mpc_result_path = os.path.join(scene_path, "MPC_following_result.png")

            if os.path.exists(mpc_result_path):
                folders_with_mpc_result += 1

                img = Image.open(mpc_result_path)
                plt.imshow(img)
                plt.title(f"Scene: {scene_token_name}")
                print(scene_token_name)
                plt.axis('off')
                plt.show()
