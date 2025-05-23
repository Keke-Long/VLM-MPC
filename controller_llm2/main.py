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
from controller_llm.fun_get_parameter_from_response import get_parameter_from_response
from controller_MPC.fun_run_mpc_controller import run_mpc_controller


# 加载nuScenes数据集
dataroot = '/home/ubuntu/Documents/Nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)


# 初始化 OpenAI 客户端
client = OpenAI(api_key="Your key")
# MODEL = "gpt-4o"
MODEL = "llava"

base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
i = 0
for idx, scene_token_name in enumerate(scene_folders):
    #print(f'idx:{idx} process {scene_token_name}')
    result_path = os.path.join(base_path, scene_token_name, 'p_VLM_MPC1')

    prompt_files = f"../scenes_data/{scene_token_name}/b_LLM_MPC/prompt.txt"
    response_folder = f"../scenes_data/{scene_token_name}/p_VLM_MPC1/gpt-4o/"
    if os.path.exists(prompt_files):
        response_file = call_llm(scene_token_name, client, MODEL, result_path, nusc)

        if glob.glob(os.path.join(result_path, MODEL, "response_*.txt")):
            i += 1
            print(i)
            get_parameter_from_response(scene_token_name, result_path, MODEL)

            # parameters_df = pd.read_csv(f"{result_path}/{MODEL}/extracted_parameters.csv", header=None)
            # parameters = dict(zip(['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway'], parameters_df.iloc[0]))
            #
            # try:
            #     run_mpc_controller(parameters, scene_token_name, f"{result_path}/{MODEL}")
            # except Exception as e:
            #     print(f"Error running MPC controller for {scene_token_name}: {e}")
            #     continue