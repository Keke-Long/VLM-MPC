import torch
import clip
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import textwrap


# 加载CLIP模型和预训练权重
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# 定义描述性文本分类
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
    "The road is a highway",
    "The road is a urban road",
    "The road is a rural road"
]

road_conditions_descriptions = [
    "The road is wet and slippery",
    "There is a construction area on the road",
    "The road is clear and safe",
    "Heavy traffic",
    "Moderate traffic",
    "Light traffic"
]

static_obstacles_descriptions = [
    "There are traffic cones ahead",
    "There is an obstacle on the road",
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
def find_best_matches(descriptions, image_features):
    text_tokens = clip.tokenize(descriptions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.cpu().numpy()

# Function to process an image and get descriptions
def clip_process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode the image using CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)

    # Find the best matching descriptions from each category
    weather_confidences = find_best_matches(weather_descriptions, image_features)
    light_confidences = find_best_matches(light_descriptions, image_features)
    road_type_confidences = find_best_matches(road_type_descriptions, image_features)
    road_conditions_confidences = find_best_matches(road_conditions_descriptions, image_features)
    static_obstacles_confidences = find_best_matches(static_obstacles_descriptions, image_features)

    # Output the results
    results = []

    # Get the best description for each category
    best_weather = weather_descriptions[weather_confidences.argmax()]
    best_light = light_descriptions[light_confidences.argmax()]
    best_road_type = road_type_descriptions[road_type_confidences.argmax()]
    best_road_conditions = road_conditions_descriptions[road_conditions_confidences.argmax()]
    best_static_obstacles = static_obstacles_descriptions[static_obstacles_confidences.argmax()]

    # Always output the best weather, light, and road type descriptions
    results.append(f"**Weather**: {best_weather}\n")
    results.append(f"**Lighting**: {best_light}\n")
    results.append(f"**Road Type**: {best_road_type}\n")

    # Only output road conditions and static obstacles if they exceed the threshold
    if max(road_conditions_confidences) >= 0.3:
        results.append(f"**Road Conditions**: {best_road_conditions}\n")
    if max(static_obstacles_confidences) >= 0.2:
        results.append(f"**Static Obstacles**: {best_static_obstacles}")

    # Collect all confidences for heatmap
    confidences = {
        'weather': weather_confidences,
        'lighting': light_confidences,
        'road_type': road_type_confidences,
        'road_conditions': road_conditions_confidences,
        'static_obstacles': static_obstacles_confidences
    }

    return results, confidences

# 加载图像
image_folder = "/home/ubuntu/Documents/Nuscenes/samples/CAM_FRONT/"
image_path = os.path.join(image_folder, 'n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915658012465.jpg')
image = Image.open(image_path)
results, confidences = clip_process_image(image_path)
print('results', results)
# Display the image and results
# plt.imshow(image)
# plt.axis('off')
# plt.title("\n".join(results))
# plt.show()

# 创建一个大图包含所有特征的子图
fig, axes = plt.subplots(5, 1, figsize=(8, 6))

cmap = 'plasma'
vmin = 0.1
vmax = 0.3

# 绘制热力图
sns.heatmap(confidences['weather'].reshape(1, -1), annot=True, cmap=cmap, ax=axes[0], vmin=vmin, vmax=vmax)
axes[0].set_title('Weather Confidences')
axes[0].set_yticks([])
axes[0].set_xticklabels(["Clear", "Cloudy", "Rainy", "Foggy", "Snowy"])

sns.heatmap(confidences['lighting'].reshape(1, -1), annot=True, cmap=cmap, ax=axes[1], cbar=False, vmin=vmin, vmax=vmax)
axes[1].set_title('Lighting Confidences')
axes[1].set_yticks([])
axes[1].set_xticklabels(["Dusk", "Night", "Daytime"])

sns.heatmap(confidences['road_type'].reshape(1, -1), annot=True, cmap=cmap, ax=axes[2], cbar=False, vmin=vmin, vmax=vmax)
axes[2].set_title('Road Type Confidences')
axes[2].set_yticks([])
axes[2].set_xticklabels(["Highway", "Urban Road", "Rural Road"])

sns.heatmap(confidences['road_conditions'].reshape(1, -1), annot=True, cmap=cmap, ax=axes[3], cbar=False, vmin=vmin, vmax=vmax)
axes[3].set_title('Road Conditions Confidences')
axes[3].set_yticks([])
axes[3].set_xticklabels(["Wet", "Construction", "Clear", "Heavy Traffic", "Moderate Traffic", "Light Traffic"])

# 换行标签
wrapped_labels = ["\n".join(textwrap.wrap(label, 10)) for label in
                  ["Traffic Cone", "Obstacle", "Accident", "Parked Veh",
                   "Animal", "Unusual Obj", "Peds", "Truck", "Workzone",
                   "Branch", "Pothole"]]
sns.heatmap(confidences['static_obstacles'].reshape(1, -1), annot=True, cmap=cmap, ax=axes[4], cbar=False, vmin=vmin, vmax=vmax)
axes[4].set_title('Static Obstacles Confidences')
axes[4].set_yticks([])
axes[4].set_xticklabels(wrapped_labels)

plt.tight_layout()
plt.savefig('clip.png', dpi=300)
