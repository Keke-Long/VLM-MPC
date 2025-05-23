import os
import shutil
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from nuscenes.nuscenes import NuScenes
import glob

from fun_generate_prompt import generate_prompt
from fun_call_llm import call_llm
from fun_get_parameter_from_response import get_parameter_from_response
from controller_MPC.fun_run_mpc_controller import run_mpc_controller


# 加载nuScenes数据集
# dataroot = '/home/ubuntu/Documents/Nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


# 初始化 OpenAI 客户端
client = OpenAI(api_key="Your key")
MODEL = "gpt-3.5-turbo"

MODEL = "llama3"
#MODEL = "llama3_no_memory"

# 读取ok_night_scenes.csv文件中的场景名称
# ok_night_scenes_path = '../calibrate_MPC/ok_night_scenes.csv'
# ok_night_scenes = pd.read_csv(ok_night_scenes_path, header=None).iloc[:, 0].tolist()


base_path ='../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
# for idx, scene_token_name in enumerate(scene_folders):
for idx, scene_token_name in enumerate(['0ac05652a4c44374998be876ba5cd6fd']):
    print(f'{idx} process {scene_token_name}')
    result_path = os.path.join(base_path, scene_token_name, 'b_LLM_MPC')

    file_path = os.path.join(base_path, scene_token_name, 'vehs_trj2.csv')
    if os.path.isfile(file_path):
        #generate_prompt(scene_token_name, result_path, nusc)
        # result_path1 = os.path.join(base_path, scene_token_name, 'b_LLM_MPC', MODEL)

        response_file = call_llm(scene_token_name, result_path, client, MODEL)
        # get_parameter_from_response(scene_token_name, result_path, MODEL)

        # parameters_df = pd.read_csv(os.path.join(result_path, MODEL, 'extracted_parameters.csv'), header=None)
        # parameters = dict(zip(['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway'], parameters_df.iloc[0]))
        # run_mpc_controller(parameters, scene_token_name, f"{result_path}/{MODEL}")

