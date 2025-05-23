import os
import pandas as pd
import numpy as np
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


def generate_updated_prompt(scene_token_name, result_path, new_result_path):
    llm2_path = os.path.join(result_path, 'llm_MPC_result.csv')
    updated_df = pd.read_csv(llm2_path)

    # 获取场景的实际开始时间
    start_timestamp = updated_df['t'].min()
    target_timestamp = start_timestamp + 5  # 5秒后的时间戳

    # 过滤掉前5秒的数据
    filtered_df = updated_df[updated_df['t'] > target_timestamp]

    # 获取最新的ego车辆状态
    ego_df = filtered_df[filtered_df['vehicle_id'] == 'ego_vehicle'].copy()
    latest_state = ego_df.iloc[ego_df['t'].idxmin()]

    # 获取最新的前车状态
    front_vehicle_df = filtered_df[filtered_df['front_vehicle'] == 1].copy()
    front_vehicle_df = front_vehicle_df.sort_values(by='t').head(1)

    if front_vehicle_df.empty:
        surrounding_vehicles = """
        No preceding vehicle detected near the current timestamp.
        """
    else:
        front_vehicle_state = front_vehicle_df.iloc[0]
        front_speed, front_acceleration = calculate_speed_acceleration(front_vehicle_df)
        surrounding_vehicles = """
        ** One vehicle is detected as the preceding vehicle:**
        - **Relative Position (delta_x, delta_y):** ({}, {})
            - **Note:** delta_y is the distance along the direction of the ego vehicle's travel, and delta_x is the distance perpendicular to the ego vehicle's travel direction.
        - **Speed (m/s):** {:.2f}
        - **Acceleration (m/s²):** {:.2f}
        """.format(
            round(front_vehicle_state['delta_y'], 2), round(front_vehicle_state['delta_x'], 2),
            front_speed.iloc[0] if not front_speed.empty else np.nan,
            front_acceleration.iloc[0] if not front_acceleration.empty else np.nan
        )

    # 计算速度和加速度（如果不存在）
    ego_speed, ego_acceleration = calculate_speed_acceleration(ego_df)
    latest_state['speed'] = ego_speed.iloc[ego_df['t'].idxmin()]
    latest_state['acceleration'] = ego_acceleration.iloc[ego_df['t'].idxmin()]

    # 读取之前的参数
    parameters_file = os.path.join(result_path, 'extracted_parameters.csv')
    parameters_df = pd.read_csv(parameters_file, header=None)
    N, Q, R, Q_h, desired_speed, desired_headway = parameters_df.iloc[0]

    # 生成新的prompt
    prompt = f"""
    Based on the previous parameters you provided, the vehicle was run for 5 seconds. Here is the updated scenario:
    
    **Updated Vehicle State:**
    - **Ego Vehicle Position (x, y):** ({round(latest_state['x'], 2)}, {round(latest_state['y'], 2)})
    - **Ego Vehicle Speed (m/s):** {latest_state['speed']:.2f}
    - **Ego Vehicle Acceleration (m/s²):** {latest_state['acceleration']:.2f}

    **Front Vehicle State:**
    {surrounding_vehicles}

    **Previous MPC Parameters:**
    - N: {N}
    - Q: {Q}
    - R: {R}
    - Q_h: {Q_h}
    - desired_speed: {desired_speed}
    - desired_headway: {desired_headway}

    Based on the updated vehicle state and the previous MPC parameters, please advise if the parameters need updating. If yes, provide the updated parameters in the following list format:

    [N, Q, R, Q_h, desired_speed, desired_headway]
    """

    prompt_file = os.path.join(new_result_path, 'prompt.txt')
    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
    with open(prompt_file, "w") as file:
        file.write(prompt)

    return prompt_file
