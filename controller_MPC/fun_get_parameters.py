import pandas as pd

# 定义 Q_h 和 desired_speed 的统计结果
parameters_info = {
    'rain': {
        'Q_h': {'P-value': 0.01889, 'Conclusion': 'Significant difference', 'Mean_0': 3.19440, 'Mean_1': 2.84978},
        'desired_speed': {'P-value': 0.20457, 'Conclusion': 'No significant difference', 'Mean': 6.43821},
    },
    'intersection': {
        'Q_h': {'P-value': 0.84493, 'Conclusion': 'No significant difference', 'Mean': 3.10500},
        'desired_speed': {'P-value': 0.03214, 'Conclusion': 'Significant difference', 'Mean_0': 6.20955, 'Mean_1': 7.09091},
    },
    'parking_lot': {
        'Q_h': {'P-value': 0.31604, 'Conclusion': 'No significant difference', 'Mean': 3.10500},
        'desired_speed': {'P-value': 0.01014, 'Conclusion': 'Significant difference', 'Mean_0': 6.20955, 'Mean_1': 7.09091},
    }
}

# 函数：根据特性获取参数
def get_parameters(scene_token_name):
    all_scenes_df = pd.read_csv('../scenes_info/all_scenes.csv')
    scene_features = all_scenes_df[all_scenes_df['scene_token'] == scene_token_name].iloc[0]

    # 如果 night feature 为 1，则使用特定参数
    if scene_features['night'] == 1:
        parameters = {
            'N': 9,
            'Q': 1,
            'R': 1.93,
            'Q_h': 3.1,
            'desired_speed': 6.4,
            'desired_headway': 2.6
        }
    else:
        parameters = {
            'N': 9,
            'Q': 1,
            'R': 1.93,
            'desired_headway': 2.6
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

