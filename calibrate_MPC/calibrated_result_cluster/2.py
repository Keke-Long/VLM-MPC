import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


# 创建结果数据框
results_data = {
    'Label': ['rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'intersection', 'intersection', 'intersection', 'intersection', 'intersection', 'intersection', 'parking_lot', 'parking_lot', 'parking_lot', 'parking_lot', 'parking_lot', 'parking_lot', 'night', 'night', 'night', 'night', 'night', 'night'],
    'MPC Parameter': ['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway', 'N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway', 'N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway', 'N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway'],
    'P-value': [0.88160, np.nan, 0.89775, 0.07155, 0.01797, 0.01300, 0.82415, np.nan, 0.32967, 0.22569, 0.07281, 0.68572, 0.15038, np.nan, 0.06134, 0.86048, 0.00048, 0.45903, 0.07312, np.nan, 0.00000, 0.00000, 0.00001, 0.13339],
    'Conclusion': ['No significant difference', 'Significant difference', 'No significant difference', 'No significant difference', 'Significant difference', 'Significant difference', 'No significant difference', 'Significant difference', 'No significant difference', 'No significant difference', 'No significant difference', 'No significant difference', 'No significant difference', 'Significant difference', 'No significant difference', 'No significant difference', 'Significant difference', 'No significant difference', 'No significant difference', 'Significant difference', 'Significant difference', 'Significant difference', 'Significant difference', 'No significant difference'],
    'Mean (group 0)': [9.14516, 1.00000, 1.68487, 2.75216, 6.43821, 2.60330, 9.14516, 1.00000, 1.68487, 2.75216, 6.01323, 2.55419, 9.14516, 1.00000, 1.68487, 2.75216, 6.43821, 2.55419, 9.14516, 1.00000, 1.92942, 3.10500, 6.43821, 2.55419],
    'Mean (group 1)': [9.14516, 1.00000, 1.68487, 2.75216, 5.09388, 2.44796, 9.14516, 1.00000, 1.68487, 2.75216, 6.01323, 2.55419, 9.14516, 1.00000, 1.68487, 2.75216, 5.09388, 2.55419, 9.14516, 1.00000, 1.15584, 1.98887, 5.09388, 2.55419]
}

results_df = pd.DataFrame(results_data)

# 创建一个新的列用于标记显著性
results_df['P-value label'] = results_df['P-value'].apply(lambda x: f'{x:.4f}*' if x < 0.05 else f'{x:.4f}')

# 设置不均匀颜色映射，中间点为0.05
norm = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)

# 使用自定义颜色映射
cmap = sns.color_palette("RdYlBu", as_cmap=True)

# 绘制热图来展示显著性结果
plt.figure(figsize=(6, 4))
pivot_table = results_df.pivot(index="MPC Parameter", columns="Label", values="P-value")

# 自定义颜色映射，添加透明度
sns.heatmap(pivot_table, annot=results_df.pivot(index="MPC Parameter", columns="Label", values="P-value label"), fmt='', cmap=cmap, cbar_kws={'label': 'P-value'}, norm=norm, linewidths=0.5, linecolor='grey', annot_kws={"size": 10, "weight": "normal", "color": "black"}, alpha=0.7)

# # 设置X轴顺序
# plt.gca().set_xticks([0.5, 1.5, 2.5, 3.5])
# plt.gca().set_xticklabels(['rain', 'night', 'intersection', 'parking_lot'])

# 调整图表以使其更紧凑
plt.tight_layout()

# 保存图表
plt.savefig('2.png', dpi=300)
