import os
import pandas as pd
import numpy as np
from nuscenes.nuscenes import NuScenes
from image_process.clip_fun import clip_process_image


# 计算速度和加速度
def calculate_speed_acceleration(df):
    # 速度
    dx = df['x'].diff().iloc[1:] / 0.5
    dy = df['y'].diff().iloc[1:] / 0.5
    speed = np.sqrt(dx ** 2 + dy ** 2)
    # 加速度
    ddx = dx.diff().iloc[1:] / 0.5
    ddy = dy.diff().iloc[1:] / 0.5
    acceleration = np.sqrt(ddx ** 2 + ddy ** 2)
    # 限制速度在0-15 m/s之间
    speed = np.clip(speed, 0, 15)
    acceleration = np.clip(acceleration, -3, 3)

    return speed, acceleration


# 定义 Q_h 和 desired_speed 的统计结果
parameters_info = {
    'rain': {
        'Q_h': {'P-value': 0.01889, 'Conclusion': 'Significant difference', 'Mean_0': 3.19, 'Mean_1': 2.85},
        'desired_speed': {'P-value': 0.20457, 'Conclusion': 'No significant difference', 'Mean': 6.43821},
    },
    'intersection': {
        'Q_h': {'P-value': 0.84493, 'Conclusion': 'No significant difference', 'Mean': 3.11},
        'desired_speed': {'P-value': 0.03214, 'Conclusion': 'Significant difference', 'Mean_0': 6.21, 'Mean_1': 7.09},
    },
    'parking_lot': {
        'Q_h': {'P-value': 0.31604, 'Conclusion': 'No significant difference', 'Mean': 3.11},
        'desired_speed': {'P-value': 0.01014, 'Conclusion': 'Significant difference', 'Mean_0': 6.21, 'Mean_1': 7.09},
    }
}


# 函数：根据特性获取参数
def get_parameters(scene_token_name):
    all_scenes_df = pd.read_csv('../scenes_info/all_scenes.csv')
    scene_features = all_scenes_df[all_scenes_df['scene_token'] == scene_token_name].iloc[0]

    parameters = {
        'N': 9,
        'Q': 1,
        'R': 1.93,
        'desired_headway': 2.60
    }
    for feature in ['rain', 'intersection', 'parking_lot']:
        for param in ['Q_h', 'desired_speed']:
            values = parameters_info[feature][param]
            if values['Conclusion'] == 'Significant difference':
                group = scene_features[feature]  # 假设 group 信息在 scene_features 中
                parameters[param] = values[f'Mean_{group}']
            else:
                parameters[param] = values['Mean']
    return parameters


# 生成 prompt 并保存为 txt 文件
def generate_prompt(scene_token_name, result_path, nusc):
    df = pd.read_csv(f"../scenes_data/{scene_token_name}/vehs_trj2.csv")
    ego_vehicle_id = 'ego_vehicle'
    ego_df = df[df['vehicle_id'] == ego_vehicle_id]

    # 获取场景信息
    dataroot = '/home/ubuntu/Documents/Nuscenes'
    scene = nusc.get('scene', scene_token_name)
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    camera_token = sample['data']['CAM_FRONT']
    camera_data = nusc.get('sample_data', camera_token)
    image_path = os.path.join(dataroot, camera_data['filename'])

    # Process the first image to get environment descriptions
    environment_descriptions, _ = clip_process_image(image_path)

    ### 获取本车和周围车信息
    # 找到最接近初始时间的前车信息
    initial_timestamp = ego_df.iloc[0]['timestamp']
    front_vehicle = df[(df['front_vehicle'] == 1) & (df['timestamp'] >= initial_timestamp + 5)].sort_values(
        by='timestamp').head(1)

    if front_vehicle.empty:
        surrounding_vehicles = """
No preceding vehicle detected near the initial timestamp.
        """
    else:
        front_vehicle_id = front_vehicle.iloc[0]['vehicle_id']
        front_vehicle_df = df[df['vehicle_id'] == front_vehicle_id].sort_values(by='timestamp')

        front_speed, front_acceleration = calculate_speed_acceleration(front_vehicle_df)

        surrounding_vehicles = """
One vehicle is detected as the preceding vehicle:
- Relative Position (delta_x, delta_y): ({}, {}) Note: delta_y is the distance along the direction of the ego vehicle's travel, and delta_x is the distance perpendicular to the ego vehicle's travel direction.
- Relative Speed (m/s): {:.2f}
- Relative Acceleration (m/s²): {:.2f}
        """.format(
            round(front_vehicle.iloc[0]['delta_y'], 2), round(front_vehicle.iloc[0]['delta_x'], 2),
            front_speed.iloc[0] if not front_speed.empty else np.nan,
            front_acceleration.iloc[0] if not front_acceleration.empty else np.nan
        )

    # 获取历史数据的平均参数
    parameters = get_parameters(scene_token_name)
    average_parameters_info = f"""
According to historical data, the average parameters in the current scenario (rain, intersection, parking_lot) are as follows:
- N: {parameters['N']}
- Q: {parameters['Q']}
- Q_h: {parameters['Q_h']}
- R: {parameters['R']}
- desired_speed: {parameters['desired_speed']}
- desired_headway: {parameters['desired_headway']}
    """

    # 生成prompt内容
    ego_speed, ego_acceleration = calculate_speed_acceleration(ego_df)

    # 提取前5秒的数据
    ego_positions = [(round(ego_df.iloc[i]['x'], 2), round(ego_df.iloc[i]['y'], 2)) for i in range(5)]
    ego_speeds = [f"{ego_speed.iloc[i]:.2f}" for i in range(5)]
    ego_accelerations = [f"{ego_acceleration.iloc[i]:.2f}" for i in range(5)]

    ego_data = f"""
Ego Vehicle State (last 5 seconds to present):
Position (x, y): {ego_positions}
Speed (m/s): [{', '.join(ego_speeds)}]
Acceleration (m/s²): [{', '.join(ego_accelerations)}]
    """

    prompt = f"""
Current Environment:
{''.join(environment_descriptions)}
{ego_data}
Surrounding Vehicles State (current moment):{surrounding_vehicles}
Average Parameters Based on Historical Data:{average_parameters_info}

The MPC parameters to be optimized include:
1. Prediction Horizon (N): The number of time steps the controller looks ahead. (Preferred value: 10)
2. Cost Weights:
   - Q (speed maintenance weight): Fixed at 1.
   - R (control effort weight): Weight for minimizing control effort.
   - Q_h (headway maintenance weight): Weight for maintaining the desired headway (distance) to the front vehicle.
3. desired_speed: The target speed (m/s) for the ego vehicle. (Range: 5 to 11 m/s)
4. desired_headway: The desired headway (s) between the ego vehicle and the front vehicle.

Step-by-Step Reasoning:
1. Understand the Current Situation.
2. Evaluate the Prediction Horizon (N).
3. Set Cost Weights:
   - Q is fixed at 1 to simplify the optimization process.
   - Balance R (control effort weight) and Q_h (headway maintenance weight) relative to Q, considering safety and comfort.
4. Define Desired Speed that is safe and efficient .
5. Determine Desired Headway to ensure a safe distance to the preceding vehicle.

Now, apply this reasoning to generate the MPC parameters.
    """

    prompt_file = f"{result_path}/prompt.txt"
    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
    with open(prompt_file, "w") as file:
        file.write(prompt)
