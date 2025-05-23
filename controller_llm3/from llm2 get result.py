import os
import shutil

base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))

for idx, scene_token_name in enumerate(scene_folders):
    scene_path = os.path.join(base_path, scene_token_name)
    result_path = os.path.join(scene_path, 'llm2')
    new_result_path = os.path.join(scene_path, 'llm3')

    # 检查 new_result_path 是否存在
    if not os.path.exists(new_result_path):
        # 复制 result_path 到 new_result_path
        shutil.copytree(result_path, new_result_path)
        print(f"Copied {result_path} to {new_result_path}")
    else:
        print(f"{new_result_path} already exists")
