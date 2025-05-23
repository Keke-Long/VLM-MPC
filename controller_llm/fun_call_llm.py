import datetime
import time
import requests
import torch



def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt


def call_gpt_model(client, MODEL, prompt):
    start_time = time.time()
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps to optimize Model Predictive Control (MPC) parameters for an autonomous driving system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    end_time = time.time()
    response1_content = response1.choices[0].message.content

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

    return response1_content, response2_content, end_time - start_time


def call_llama_model(MODEL, prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "max_tokens": 200,
            "temperature": 0,
            "top_k": 50,
            "device": "cuda"
        },
        "return_attention": True
    }
    start_time = time.time()
    response = requests.post(url, json=data)
    response1_content = response.json()['response']
    result = response.json()
    print(result)  # 打印完整的响应以调试
    attention_matrices = result.get('attention')
    end_time = time.time()
    print(attention_matrices)

    # 第二步：让llama3按照特定格式返回结果
    data = {
        "model": MODEL,
        "prompt": f"{prompt}\n{response1_content}\nPlease provide the MPC parameters in the following list format: [N, Q, R, Q_h, desired_speed, desired_headway].",
        "stream": False,
        "options": {
            "max_tokens": 20,  # 可以根据需要调整
            "temperature": 0,
            "device": "cuda"
        }
    }
    response = requests.post(url, json=data)
    response2_content = response.json()['response']

    return response1_content, response2_content, end_time - start_time


def call_llm(scene_token_name, result_path, client, MODEL):
    # 读取prompt
    prompt = read_txt_file(f"{result_path}/prompt.txt")

    if MODEL.startswith("gpt"):
        response1_content, response2_content, duration = call_gpt_model(client, MODEL, prompt)
    elif MODEL == "llama3":
        response1_content, response2_content, duration = call_llama_model(MODEL, prompt)

    # 保存两次响应结果到同一个文件
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    response_file = f"{result_path}/{MODEL}/response_{current_time}.txt"
    with open(response_file, "w") as file:
        file.write(response1_content)
        file.write('\n')
        file.write(response2_content)

    # 记录时间到文件
    print('duration = ', duration)
    # time_file = f"{result_path}/{MODEL}/responsetime.txt"
    # with open(time_file, "a") as file:
    #     file.write(f"{duration}\n")

    return response_file