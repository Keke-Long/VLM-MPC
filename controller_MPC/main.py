import os
import pandas as pd
from fun_get_parameters import get_parameters
from fun_run_mpc_controller import run_mpc_controller


# # 夜间平均值
# parameters = {
#     'N': 9,
#     'Q': 1,
#     'R': 1.93,
#     'Q_h': 3.2,  # 你可以根据具体场景修改这些值
#     'desired_speed': 6.4,  # 同样，你可以根据具体场景修改这些值
#     'desired_headway': 2.6
# }

# 读取ok_night_scenes.csv文件中的场景名称
# ok_night_scenes_path = '../calibrate_MPC/ok_night_scenes.csv'
# ok_night_scenes = pd.read_csv(ok_night_scenes_path, header=None).iloc[:, 0].tolist()


base_path = '../scenes_data/'
scene_folders = sorted(os.listdir(base_path))
# for scene_token_name in scene_folders:
for scene_token_name in [                        'b07358651c604e2d83da7c4d4755de73'
                         ]:
    print(f'process {scene_token_name}')
    calibrated_MPC_parameter_path = f"{base_path}{scene_token_name}/MPC_calibration/best_MPC_parameter.csv"
    if os.path.isfile(calibrated_MPC_parameter_path):
        result_path = f"{base_path}{scene_token_name}/b_MPC/"
        os.makedirs(result_path, exist_ok=True)

        parameters = get_parameters(scene_token_name)
        parameters_list = [parameters['N'], parameters['Q'], parameters['R'], parameters['Q_h'], parameters['desired_speed'], parameters['desired_headway']]
        run_mpc_controller(parameters, scene_token_name, result_path)


# '28e4134bcd664522907277f1ceec2893'
# '2bbae5e654224cbeb5884c471e2ad05e',
# '2fc3753772e241f2ab2cd16a784cc680',
# '400b708de5d7434d8964da4a0246d164',
# '5560a973257e407b8d5bf9fb92b1e0f3',
# '567fcd99d0dc4fa088f52b047f4ebdcf',
# '5aa5225d0ef440519a31c1dde075dab7',
# '6498fce2f38645fc9bf9d4464b159230',
# '69e393da7cb54eb6bea7927a2af0d1ee',
# '6d6314f1865343f987ca53107b694745',
# '7ca5e90766dc4b7bb6a9b3b0d95c99da',
# '7ed1b613f0ee47ceaf6d19ca3947f825',
# '8e3364691ee94c698458e0913a29af78',
# 'c9d654d4279c4efa9745a25ba6b373ee',
# 'cab845a2e8864482993ac62811f879c1',
# 'cd9db61edff14e8784678abb347cd674',
# 'd01e7279da2649ef896dc42f6b9ee7ab',
# 'd3c39710e9da42f48b605824ce2a1927',
# 'e8834785d9ff4783a5950281a4579943',
# 'fb3aaad97849430cbc32891319c9be10'