import pandas as pd
import numpy as np
from shapely.geometry import Point
from nuscenes.map_expansion.map_api import NuScenesMap

def stop_line(scene_token_name, nusc):

    # 加载vehs_trj.csv数据
    csv_path = f'../scenes_data1/{scene_token_name}/vehs_trj2.csv'
    vehs_trj_df = pd.read_csv(csv_path)

    # 提取自车轨迹
    ego_df = vehs_trj_df[vehs_trj_df['vehicle_id'] == 'ego_vehicle'].copy()

    # 获取scene的地图信息
    dataroot = '/home/ubuntu/Documents/Nuscenes'
    log = nusc.get('log', nusc.get('scene', scene_token_name)['log_token'])
    map_name = log['location']
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)

    # 检查ego pose是否经过stop_line
    stop_lines = nusc_map.stop_line

    def check_stop_line(ego_x, ego_y):
        point = Point(ego_x, ego_y)  # 将pose转换为Shapely的Point对象

        for stop_line in stop_lines:
            polygon_token = stop_line['polygon_token']
            polygon = nusc_map.extract_polygon(polygon_token)

            if polygon.contains(point):
                return stop_line['token'], stop_line['stop_line_type']
        return None, None

    # 初始化新的列
    ego_df['stop_line_token'] = None
    ego_df['stop_line_need_stop'] = None

    # 遍历自车轨迹数据，检查每个时间点是否位于stop_line内
    for idx in range(len(ego_df)):
        ego_x = ego_df.iloc[idx]['x']
        ego_y = ego_df.iloc[idx]['y']
        ego_timestamp = ego_df.iloc[idx]['timestamp']

        stop_line_token, stop_line_type = check_stop_line(ego_x, ego_y)

        if stop_line_token:
            ego_df.loc[idx, 'stop_line_token'] = stop_line_token

    # 计算速度
    ego_df['speed'] = 0
    for idx in range(1, len(ego_df) - 1):
        dist_prev = np.linalg.norm([ego_df.iloc[idx]['x'] - ego_df.iloc[idx - 1]['x'], ego_df.iloc[idx]['y'] - ego_df.iloc[idx - 1]['y']])
        dist_next = np.linalg.norm([ego_df.iloc[idx + 1]['x'] - ego_df.iloc[idx]['x'], ego_df.iloc[idx + 1]['y'] - ego_df.iloc[idx]['y']])
        time_diff_prev = ego_df.iloc[idx]['timestamp'] - ego_df.iloc[idx - 1]['timestamp']
        time_diff_next = ego_df.iloc[idx + 1]['timestamp'] - ego_df.iloc[idx]['timestamp']

        speed_prev = dist_prev / time_diff_prev if time_diff_prev > 0 else 0
        speed_next = dist_next / time_diff_next if time_diff_next > 0 else 0

        ego_df.loc[idx, 'speed'] = (speed_prev + speed_next) / 2

    # 确定哪些stop_line需要停车
    stop_lines_to_check = ego_df['stop_line_token'].dropna().unique()

    for stop_line_token in stop_lines_to_check:
        stop_line_df = ego_df[ego_df['stop_line_token'] == stop_line_token]
        slow_speeds = stop_line_df[stop_line_df['speed'] < 0.5]

        if len(slow_speeds) >= 3:
            ego_df.loc[ego_df['stop_line_token'] == stop_line_token, 'stop_line_need_stop'] = 1
        else:
            ego_df.loc[ego_df['stop_line_token'] == stop_line_token, 'stop_line_need_stop'] = 0

    # 合并ego_df和原始数据框，只更新ego_vehicle的行
    vehs_trj_df = vehs_trj_df.merge(ego_df[['timestamp', 'stop_line_token', 'stop_line_need_stop']], on='timestamp', how='left')
    vehs_trj_df.loc[vehs_trj_df['vehicle_id'] != 'ego_vehicle', ['stop_line_token', 'stop_line_need_stop']] = None

    # 保存更新后的CSV文件
    vehs_trj_df.to_csv(csv_path, index=False)
