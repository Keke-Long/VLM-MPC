import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
scene_token_name = '02f1e5e2fc544798aad223f5ae5e8440'
data = pd.read_csv(f"../../scenes_data/{scene_token_name}/vehs_trj_MPC_result.csv")

# 只考虑 'ego_vehicle'
ego_data = data[data['vehicle_id'] == 'ego_vehicle']

# Function to calculate speeds between consecutive points for a given set of coordinates
def calculate_speeds(x, y):
    speeds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)/0.5  # Speeds calculated between consecutive points
    speeds_padded = np.append(speeds, np.nan)  # Pad with NaN for the last point
    return speeds_padded

# Calculate speeds for both original and new trajectories
ego_data['speed'] = calculate_speeds(ego_data['x'], ego_data['y'])
ego_data['speed_new'] = calculate_speeds(ego_data['x_new'], ego_data['y_new'])
print(ego_data['x_new'])

# Function to calculate accelerations from speeds
def calculate_accelerations(speeds):
    accelerations = np.diff(speeds)  # Accelerations calculated as the difference in consecutive speeds
    accelerations_padded = np.append(accelerations, np.nan)  # Pad with NaN for the last point
    return accelerations_padded

# Calculate accelerations for both original and new trajectories
ego_data['acceleration'] = calculate_accelerations(ego_data['speed'].values)
ego_data['acceleration_new'] = calculate_accelerations(ego_data['speed_new'].values)

# Plotting the distributions
fig, axs = plt.subplots(1, 2, figsize=(8, 3))


# Extract min and max values for original and new trajectories for speed
speed_orig_min, speed_orig_max = ego_data['speed'].min(), ego_data['speed'].max()
speed_new_min, speed_new_max = ego_data['speed_new'].min(), ego_data['speed_new'].max()

# Extract min and max values for original and new trajectories for acceleration
accel_orig_min, accel_orig_max = ego_data['acceleration'].min(), ego_data['acceleration'].max()
accel_new_min, accel_new_max = ego_data['acceleration_new'].min(), ego_data['acceleration_new'].max()

# Combined speed distributions with detailed range in title
axs[0].hist(ego_data['speed'].dropna(), bins=50, color='blue', alpha=0.5, edgecolor='black', label='real-world')
axs[0].hist(ego_data['speed_new'].dropna(), bins=50, color='green', alpha=0.5, edgecolor='black', label='LLM_MPC')
axs[0].set_title(f'Speed Distribution\nOriginal: {speed_orig_min:.2f} - {speed_orig_max:.2f}\nMPC: {speed_new_min:.2f} - {speed_new_max:.2f}')
axs[0].set_xlabel('Speed (m/s)')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Combined acceleration distributions with detailed range in title
axs[1].hist(ego_data['acceleration'].dropna(), bins=50, color='red', alpha=0.5, edgecolor='black', label='real-wrold')
axs[1].hist(ego_data['acceleration_new'].dropna(), bins=50, color='orange', alpha=0.5, edgecolor='black', label='LLM_MPC')
axs[1].set_title(f'Acceleration Distribution\nOriginal: {accel_orig_min:.2f} - {accel_orig_max:.2f}\nMPC: {accel_new_min:.2f} - {accel_new_max:.2f}')
axs[1].set_xlabel('Acceleration (m/$s^2$)')
axs[1].set_ylabel('Frequency')
axs[1].legend()

plt.tight_layout()
plt.savefig(f"../../scenes_data/{scene_token_name}/MPC_compare_v_a.png")
plt.show()