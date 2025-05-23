import os
import re
import csv
import glob


# def extract_mpc_parameters(response_text):
#     # 定义正则表达式来匹配参数，包括带有单位的情况
#     pattern = r"\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*m/s,\s*(\d+\.?\d*)\s*s\s*\]"
#     # 直接从全文中匹配参数
#     match = re.search(pattern, response_text)
#     if match:
#         # 去掉单位并将字符串转换为浮点数
#         parameters = [float(match.group(1)), float(match.group(2)), float(match.group(3)),
#                       float(match.group(4)), float(match.group(5)), float(match.group(6))]
#         return parameters
#
#     # 尝试匹配不带单位的情况
#     pattern_no_units = r"\[\s*(\d+),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\]"
#     match_no_units = re.search(pattern_no_units, response_text)
#     if match_no_units:
#         parameters = list(map(float, match_no_units.groups()))
#         return parameters
#     return None


def extract_mpc_parameters(response_text):
    # 定义正则表达式来匹配包含6个数字的方括号，忽略中间的单位或其他文字
    pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*\]"

    # 尝试匹配包含单位或其他文字的情况
    match = re.search(pattern, response_text)
    if match:
        # 将匹配到的参数去掉单位并转换为浮点数
        parameters = [float(match.group(i)) for i in range(1, 7)]
        return parameters

    return None


def save_parameters_to_csv(parameters, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(parameters)


def get_parameter_from_response(scene_token_name, result_path, MODEL):
    response_files = glob.glob(os.path.join(result_path, MODEL, "response_*.txt"))
    # 找到最新修改的文件
    latest_file = max(response_files, key=os.path.getmtime)

    with open(latest_file, 'r') as file:
        response_text = file.read()
    parameters = extract_mpc_parameters(response_text)
    # print('parameters', parameters)

    output_csv = f"{result_path}/{MODEL}/extracted_parameters.csv"
    if parameters:
        pass
        # save_parameters_to_csv(parameters, output_csv)
    else:
        print(scene_token_name, 'Can NOT get parameters')