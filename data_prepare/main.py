import pandas as pd
from nuscenes.nuscenes import NuScenes

from step1_get_trj import get_trj
from step1_plot_all_trj import plot_all_trj
from step2_cal_realtive_position import cal_relative_position
from step3_find_front_vehicle import find_front_vehicle
from step4_plot_front_vehicle import plot_front_vehicle
from step5_stop_line import stop_line

from plot_cam_front_pic import plot_cam_front_pic


def read_and_assign_values(scene_token_name, nusc):
    # get_trj(scene_token_name, nusc) # 得到"vehs_trj.csv" 存的是没有处理的所有车的数据
    # plot_all_trj(scene_token_name)

    cal_relative_position(scene_token_name, nusc) # 得到 "vehs_trj2.csv" 所有附加信息都加到这个文件
    find_front_vehicle(scene_token_name)
    plot_front_vehicle(scene_token_name)
    stop_line(scene_token_name, nusc)

    # plot_cam_front_pic(scene_token_name, nusc)


# 处理所有scenes
if __name__ == "__main__":
    # 配置nuScenes数据集路径和版本
    dataroot = '/home/ubuntu/Documents/Nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

    # df = pd.read_csv('../scenes_info/boston_scenes.csv')
    # filtered_df = df[(df['turn'] == 0) & (df['checked_turn'] == 0) & (df['length_more_than_10'] == 1)]
    # for scene_token_name in filtered_df['scene_token']:

    df = pd.read_csv('../scenes_info/night_scenes.csv')
    for scene_token_name in df['scene_token']:

    #for scene_token_name in ['55638ae3a8b34572aef756ee7fbce0df']:
        print(f'scene_token_name={scene_token_name}')
        read_and_assign_values(scene_token_name, nusc)
