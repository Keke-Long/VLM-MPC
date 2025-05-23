import os
import shutil
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from nuscenes.nuscenes import NuScenes
import glob
from fun_generate_prompt import generate_updated_prompt
from fun_call_llm import call_llm
from controller_llm.fun_get_parameter_from_response import get_parameter_from_response
from fun_run_mpc_controller import run_mpc_controller


# 初始化 OpenAI 客户端
client = OpenAI(api_key="Your key")
MODEL = "llava"


# 加载nuScenes数据集
# dataroot = '/home/ubuntu/Documents/Nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
for idx, scene_token_name in enumerate(scene_folders):
    print(f'{idx} process {scene_token_name}')
    scene_path = os.path.join(base_path, scene_token_name)
    if os.path.isdir(scene_path):
        result_path = os.path.join(scene_path, 'p_VLM_MPC1')
        new_result_path = os.path.join(scene_path, 'p_VLM_MPC2')

        os.makedirs(new_result_path, exist_ok=True)

        # prompt_files = glob.glob(os.path.join(result_path, 'prompt.txt'))
        # if not prompt_files:
        # generate_updated_prompt(scene_token_name, result_path, new_result_path)

        # response_files = glob.glob(os.path.join(new_result_path, 'response*.txt'))
        # if not response_files:
        # response_file = call_llm(scene_token_name, client, MODEL, new_result_path)

        parameter_files = glob.glob(os.path.join(result_path, MODEL, 'extracted_parameters.csv'))
        # if not parameter_files:
        get_parameter_from_response(scene_token_name, new_result_path, MODEL)

        #run_mpc_controller(scene_token_name, f"{new_result_path}/{MODEL}")