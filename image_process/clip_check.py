'''
检验clip的结果，遍历图片检查没，
'''

import torch
import clip
import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# 加载CLIP模型和预训练权重
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图像
image_folder = "/home/ubuntu/Documents/Nuscenes/samples/CAM_FRONT/" \

# 获取文件夹中的所有图像文件
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])[350:400]


# Define candidate text descriptions categorized by type
weather_descriptions = [
    "The weather is clear",
    "The weather is cloudy",
    "The weather is rainy",
    "The weather is foggy",
    "The weather is snowy"
]

light_descriptions = [
    "Dusk",
    "Night",
    "Daytime"
]

road_type_descriptions = [
    "The road is highway",
    "The road is main road",
    "The road is rural road",
    "The road is commercial area road",
    "The road is near a school zone",
    "The road is inside parking_lot"
]

road_conditions_descriptions = [
    "The road is wet and slippery",
    "There is a construction area on the road",
    "The road is clear and safe",
    "The road is a highway",
    "The road is a main road",
    "The road is a rural road",
    "The road is a commercial area road",
    "The road is near a school zone",
    "Heavy traffic",
    "Moderate traffic",
    "Light traffic"
]

static_obstacles_descriptions = [
    "There are traffic cones ahead",
    "There is an obstacle on the road",
    "There is a traffic sign ahead",
    "There is an accident ahead",
    "There are parked vehicles on the roadside",
    "There are animals on the road",
    "There are unusual objects on the road",
    "Pedestrians crossing",
    "Truck ahead",
    "Workzones ahead",
    "Branches on the road",
    "Potholes on the road"
]

# Function to find the best matching description from a category with confidence threshold
def find_best_match(descriptions, image_features, threshold=0.1):
    text_tokens = clip.tokenize(descriptions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        similarity = torch.cosine_similarity(image_features, text_features)
        best_match_index = similarity.argmax().item()
        best_match_value = similarity[best_match_index].item()
    if best_match_value >= threshold:
        return descriptions[best_match_index], best_match_value
    else:
        return None, best_match_value


# Function to process an image and get descriptions
def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode the image using CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)

    # Find the best matching descriptions from each category
    best_weather, weather_confidence = find_best_match(weather_descriptions, image_features)
    best_light, light_confidence = find_best_match(light_descriptions, image_features)
    best_road_type, road_type_confidence = find_best_match(road_type_descriptions, image_features)
    best_road_conditions, road_conditions_confidence = find_best_match(road_conditions_descriptions, image_features)
    best_static_obstacles, static_obstacles_confidence = find_best_match(static_obstacles_descriptions, image_features)

    # Output the results
    results = []

    # Always output the best weather and light descriptions
    results.append(f"Weather: {best_weather}")
    results.append(f"Lighting: {best_light}")
    results.append(f"Road Type: {best_road_type}")

    # Only output road conditions and static obstacles if they exceed the threshold
    if road_conditions_confidence >= 0.3:
        results.append(f"Road Conditions: {best_road_conditions}")
    if static_obstacles_confidence >= 0.2:
        results.append(f"Static Obstacles: {best_static_obstacles}")

    return results


# 遍历图像文件夹中的前100张图像
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    results = process_image(image_path)

    # Display the image and results
    plt.imshow(image)
    plt.axis('off')
    plt.title("\n".join(results))
    plt.show()