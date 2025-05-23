import datetime
import os
import shutil
import base64


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt


def call_llm(scene_token_name, client, MODEL, result_path):
    # 第一步：获取推理结果
    prompt_file = os.path.join(result_path, 'prompt.txt')
    prompt = read_txt_file(prompt_file)

    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are an AI assistant that helps to optimize Model Predictive Control (MPC) parameters for an autonomous driving system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response1_content = response1.choices[0].message.content

    # 第二步：让GPT按照特定格式返回结果
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps to optimize Model Predictive Control (MPC) parameters for an autonomous driving system."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response1_content},
            {"role": "user", "content": "Please provide the MPC parameters in the following list format: [N, Q, R, Q_h, desired_speed, desired_headway]."}
        ],
        temperature=0
    )

    response2_content = response2.choices[0].message.content

    # 保存两次响应结果到同一个文件
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    response_file = os.path.join(result_path, f'response_{current_time}.txt')
    with open(response_file, "w") as file:
        file.write(response1_content)
        file.write('\n')
        file.write(response2_content)

    return response_file