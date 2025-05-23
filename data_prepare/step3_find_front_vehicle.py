'''
根据相对位置 标记前车
保存前车信息到csv
'''

import csv
import pandas as pd


def get_front_object_detections(data_dict, ego_vehicle_id):
    objects = data_dict["objects"]
    detected_objs = []
    min_distance = float('inf')
    closest_obj = None
    for obj in objects:
        if obj['id'] == ego_vehicle_id:
            continue  # 排除本车
        # search for the front objects
        obj_y, obj_x = obj["bbox"][:2]
        if abs(obj_x) < 2.0 and obj_y >= 0.0 and obj_y < 200.0:
            distance = obj_y
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
    if closest_obj:
        detected_objs.append(closest_obj)
    return detected_objs


def find_front_vehicle(scene_token_name):
    # 加载CSV文件
    csv_path = f'../scenes_data1/{scene_token_name}/vehs_trj2.csv'
    df = pd.read_csv(csv_path)

    # 确定本车ID（假设本车ID为第一个时间点的第一个车辆）
    ego_vehicle_id = df.loc[df['timestamp'] == df['timestamp'].min(), 'vehicle_id'].values[0]

    # 计算每个时间点的前车信息
    timestamps = df['timestamp'].unique()
    front_vehicle_data = []

    for timestamp in timestamps:
        # 提取当前时间点的所有车辆数据
        current_data = df[df['timestamp'] == timestamp]
        objects = []
        for _, row in current_data.iterrows():
            objects.append({
                "id": row['vehicle_id'],
                "name": "vehicle",
                "bbox": [row['delta_x'], row['delta_y']]
            })
        data_dict = {"objects": objects}
        detected_objs = get_front_object_detections(data_dict, ego_vehicle_id)

        # 标记前车
        for obj in objects:
            obj_id = obj["id"]
            is_front_vehicle = 1 if obj in detected_objs else 0
            front_vehicle_data.append([obj_id, timestamp, is_front_vehicle])

    # 将前车信息添加到原始数据中
    front_vehicle_df = pd.DataFrame(front_vehicle_data, columns=['vehicle_id', 'timestamp', 'front_vehicle'])
    df = df.merge(front_vehicle_df, on=['vehicle_id', 'timestamp'], suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')

    # 保存更新后的CSV文件
    df.to_csv(csv_path, index=False)

    print(f"Front vehicle saved")



if __name__ == "__main__":
    scene_token_name = 'e5a3df5fe95149b5b974af1d14277ea7'
    find_front_vehicle(scene_token_name)
