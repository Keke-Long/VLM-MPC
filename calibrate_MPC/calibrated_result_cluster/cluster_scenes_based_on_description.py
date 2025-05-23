import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 定义参数的名称
parameter_names = ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway']
label_columns = ['rain', 'pedestrian', 'intersection', 'lane_change', 'parking_vehicles', 'parking_lot']

# 初始化一个空的数据框架，用于存储所有场景的最佳参数
all_params = pd.DataFrame(columns=parameter_names)
all_labels = pd.DataFrame(columns=label_columns)

# 加载描述数据
df_description = pd.read_csv('../../scenes_info/all_scenes.csv')

# 遍历 ../../scenes_data/ 文件夹中的每个子文件夹
base_path = '../../scenes_data1/'
for scene_token_name in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_token_name)
    if os.path.isdir(scene_path):
        best_param_file_path = os.path.join(scene_path, 'best_parameter.csv')
        if os.path.exists(best_param_file_path):
            # 读取 best_parameter.csv 文件，手动指定列名
            df = pd.read_csv(best_param_file_path, header=None, names=parameter_names)
            if not df.empty:  # 检查文件是否为空
                last_row = df.iloc[[-1]]  # 这里使用双括号来确保返回的是 DataFrame
                all_params = pd.concat([all_params, last_row], ignore_index=True)

                # 读取对应的label_columns这些01变量
                labels = df_description[df_description['scene_token'] == scene_token_name][label_columns]
                if not labels.empty:
                    all_labels = pd.concat([all_labels, labels], ignore_index=True)

# 处理MPC参数数据
all_params = all_params.apply(pd.to_numeric, errors='coerce')
all_params = all_params.dropna()
scaler = StandardScaler()
scaled_params = scaler.fit_transform(all_params)

# 合并标签和标准化后的MPC参数
combined_features = np.hstack((all_labels, scaled_params))

# 聚类分析
num_clusters = 2  # 设定聚类数目
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(combined_features)

# 将聚类结果添加到all_params中
all_params['Cluster'] = clusters

# 创建一个新的数据框架来存储每个集群的标签分布和参数平均值
cluster_summary = pd.DataFrame(columns=['Cluster'] + label_columns + parameter_names)

# 提取每个集群的标签分布和参数平均值
for cluster in range(num_clusters):
    cluster_data = all_params[all_params['Cluster'] == cluster]

    # 计算标签的平均值（即每个标签的比例）
    cluster_labels = all_labels.iloc[cluster_data.index]
    cluster_label_means = cluster_labels.mean().to_dict()

    # 计算MPC参数的平均值
    cluster_mpc_means = cluster_data[parameter_names].mean().to_dict()

    # 添加到数据框架中
    cluster_summary = pd.concat([cluster_summary, pd.DataFrame([{
        'Cluster': cluster,
        **cluster_label_means,
        **cluster_mpc_means
    }])], ignore_index=True)

# 保存cluster_summary为CSV文件
cluster_summary.to_csv('cluster_summary.csv', index=False)
print(cluster_summary)

# 可视化聚类结果
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of MPC Parameters by Cluster')

for i, param in enumerate(parameter_names):
    row = i // 3
    col = i % 3
    for cluster in range(num_clusters):
        cluster_data = all_params[all_params['Cluster'] == cluster]
        axs[row, col].hist(cluster_data[param], bins=10, alpha=0.5, label=f'Cluster {cluster}')
    axs[row, col].set_title(param)
    axs[row, col].set_xlabel('Value')
    axs[row, col].set_ylabel('Frequency')

plt.legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()