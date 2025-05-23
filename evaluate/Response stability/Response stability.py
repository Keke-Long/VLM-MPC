import os
import re
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd


def extract_mpc_parameters(response_text):
    # 定义正则表达式来匹配参数
    pattern_list = [
        r"\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\]",
        r"\\\[(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\\\]",
        r"\[(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]",
        r"\[N, Q, R, Q_h, desired_speed, desired_headway\]\s*=\s*\[(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]",
        r"\*\*Prediction Horizon \(N\):\*\*\s*(\d+)",
        r"\*\*Q \(speed maintenance weight\):\*\*\s*(\d+\.?\d*)",
        r"\*\*R \(control effort weight\):\*\*\s*(\d+\.?\d*)",
        r"\*\*Q_h \(headway maintenance weight\):\*\*\s*(\d+\.?\d*)",
        r"\*\*Desired Speed:\*\*\s*(\d+\.?\d*)\s*m/s",
        r"\*\*Desired Headway:\*\*\s*(\d+\.?\d*)\s*seconds",
        r"\[MPC Parameters:\s*\[N,\s*Q,\s*R,\s*Q_h,\s*desired_speed,\s*desired_headway\]\]\s*\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\]",
        r"\[MPC Parameters:\s*\[N,\s*Q,\s*R,\s*Q_h,\s*desired_speed,\s*desired_headway\]\]\n\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\]",
        r"\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]",
        r"\s*\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\]",
        r"\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*"
    ]

    # 尝试从最后一行获取参数
    last_line = response_text.splitlines()[-1]
    for pattern in pattern_list:
        match = re.search(pattern, last_line)
        if match:
            parameters = list(map(float, match.groups()))
            return parameters

    # 如果最后一行没有找到，遍历全文
    parameters = []
    for pattern in pattern_list[1:]:
        match = re.search(pattern, response_text)
        if match:
            parameters.extend(match.groups())

    if len(parameters) == 6:
        return list(map(float, parameters))
    return None


def save_parameters_to_csv(parameters, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(parameters)


def get_parameter_from_response(scene_token_name, result_path, model):
    response_files = glob.glob(os.path.join(result_path, model, "response_*.txt"))
    all_parameters = []

    # 确保输出文件存在且为空
    output_file = f"{model}_parameters.csv"
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway'])

    # 遍历所有 response 文件
    for file_path in response_files:
        with open(file_path, 'r') as file:
            response_text = file.read()
            parameters = extract_mpc_parameters(response_text)
            if parameters:
                all_parameters.append(parameters)
                #save_parameters_to_csv(parameters, output_file)

    # 绘制参数分布直方图
    if all_parameters:
        df = pd.DataFrame(all_parameters, columns=['N', 'Q', 'R', 'Q_h', 'desired_speed', 'desired_headway'])
        fig, axes = plt.subplots(1, 5, figsize=(7, 2))  # 创建一行五列的子图

        # 绘制每个参数的直方图
        df['N'].hist(ax=axes[0], bins=10, color='lightseagreen', alpha=0.7)
        axes[0].set_xlabel('$N$')
        axes[0].set_ylabel('Frequency')

        df['R'].hist(ax=axes[1], bins=10, color='lightseagreen', alpha=0.7)
        axes[1].set_xlabel('$R$')

        df['Q_h'].hist(ax=axes[2], bins=10, color='lightseagreen', alpha=0.7)
        axes[2].set_xlabel('$Q^h$')

        df['desired_speed'].hist(ax=axes[3], bins=10, color='lightseagreen', alpha=0.7)
        axes[3].set_xlabel('$v^d$')

        df['desired_headway'].hist(ax=axes[4], bins=10, color='lightseagreen', alpha=0.7)
        axes[4].set_xlabel('$h^d$')

        for i in range(5):
            ax = axes[i]
            ax.grid(False)
        plt.tight_layout()
        # plt.savefig(f"{scene_token_name}_parameter_distributions.png")
        plt.show()



scene_token_name = "0ac05652a4c44374998be876ba5cd6fd"


result_path = f"/home/ubuntu/Documents/VLM_MPC/scenes_data/{scene_token_name}/p_VLM_MPC1/"

MODEL = "gpt-4o"  # 结果会变
# MODEL = "llama3"  # 结果不会变

get_parameter_from_response(scene_token_name, result_path, MODEL)

