import datetime
import time
import os
import shutil
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re
import torch
import requests
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig



def read_txt_file(file_path, Memory = True, environment=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if not Memory:
        start_marker = "Average Parameters Based on Historical Data:"
        end_marker = "The MPC parameters to be optimized include:"
        # Use regular expression to remove the section between the start and end markers
        pattern = re.compile(r'%s.*?(?=%s)' % (re.escape(start_marker), re.escape(end_marker)), re.DOTALL)
        content = re.sub(pattern, '', content)

    if not environment:
        env_marker = "Ego Vehicle State (last 5 seconds to present):"
        # Use regular expression to remove everything before the environment marker
        pattern = re.compile(r'.*?(?=%s)' % re.escape(env_marker), re.DOTALL)
        content = re.sub(pattern, '', content)

    return content


# 提取车辆前方摄像头的第一张照片
def get_first_front_camera_image(scene_token_name, nusc):
    # 获取场景信息
    scene = nusc.get('scene', scene_token_name)
    sample_token = scene['first_sample_token']
    # 获取第一张前方摄像头图片路径
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_filepath = nusc.get_sample_data_path(cam_token)
    return cam_filepath


def get_first_five_front_camera_image(scene_token_name, nusc):
    # 获取场景信息
    scene = nusc.get('scene', scene_token_name)
    sample_token = scene['first_sample_token']
    images = []
    for i in range(5):
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_filepath = nusc.get_sample_data_path(cam_token)
        images.append(cam_filepath)
        # 获取下一个sample token
        sample_token = sample['next']
        if not sample_token:
            break
    return images


# 将图像转换为base64编码
def image_to_base64(image_data):
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return base64_image


def compress_image(image_path, quality=100):
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def call_gpt_model(client, MODEL, prompt, base64_image):
    start_time = time.time()
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps to optimize Model Predictive Control (MPC) parameters for an autonomous driving system. "
                                          "Please note that we will call the AI model again in the next 20 seconds with updated scene information. The MPC parameters generated should be considered for the short term and may need to be updated."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        temperature=0
    )
    end_time = time.time()
    response1_content = response1.choices[0].message.content

    # 第二步：让GPT按照特定格式返回结果
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps to optimize Model Predictive Control (MPC) parameters for an autonomous driving system. "
                                          "Please note that we will call the AI model again in the next 20 seconds with updated scene information. The MPC parameters generated should be considered for the short term and may need to be updated."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response1_content},
            {"role": "user", "content": "Please provide the MPC parameters in the following list format: [N, Q, R, Q_h, desired_speed, desired_headway]."}
        ],
        temperature=0
    )

    response2_content = response2.choices[0].message.content
    return response1_content, response2_content, end_time - start_time


def call_llama_model(MODEL, prompt, base64_image):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": MODEL,
        "prompt": f"{prompt}\n![image](data:image/jpeg;base64,{base64_image})",
        "stream": False,
        "options": {
            "max_tokens": 200,
            "temperature": 0,
            "top_k": 20
        }
    }
    start_time = time.time()
    response = requests.post(url, json=data)
    response1_content = response.json()['response']
    end_time = time.time()

    # 第二步：让llama3按照特定格式返回结果
    data = {
        "model": MODEL,
        "prompt": f"{prompt}\n{response1_content}\nPlease provide the MPC parameters in the following list format: [N, Q, R, Q_h, desired_speed, desired_headway].",
        "stream": False,
        "options": {
            "max_tokens": 20,
            "temperature": 0
        }
    }

    response = requests.post(url, json=data)
    response2_content = response.json()['response']

    return response1_content, response2_content, end_time - start_time


def call_llm(scene_token_name, client, MODEL, result_path, nusc):
    # 第一步：获取推理结果
    prompt = read_txt_file(f"../scenes_data/{scene_token_name}/b_LLM_MPC/prompt.txt")
    prompt += "\nAdditional Information:" \
              "\n- We have attached one snapshots of the scene from the ego vehicle's perspective, which corresponds to Ego Vehicle State at present"

    # 获取车辆前方摄像头的照片
    image_path = get_first_front_camera_image(scene_token_name, nusc)
    base64_image = image_to_base64(compress_image(image_path, quality=100))

    if MODEL.startswith("gpt"):
        response1_content, response2_content, duration = call_gpt_model(client, MODEL, prompt, base64_image)
    elif MODEL in ["llama3", "llava"]:
        response1_content, response2_content, duration = call_llama_model(MODEL, prompt, image_path)
    else:
        raise ValueError(f"Unsupported model: {MODEL}")

    # 保存两次响应结果到同一个文件
    os.makedirs(f"{result_path}/{MODEL}", exist_ok=True)

    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    response_file = f"{result_path}/{MODEL}/response_{current_time}.txt"
    with open(response_file, "w") as file:
        file.write(response1_content)
        file.write('\n')
        file.write(response2_content)

    # 记录时间到文件
    print('duration = ', duration)
    time_file = f"{result_path}/{MODEL}/responsetime.txt"
    with open(time_file, "a") as file:
        file.write(f"{duration}\n")

    return response_file