import torch
import clip
from PIL import Image

# 加载CLIP模型和预训练权重
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

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
def clip_process_image(image_path):
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

    # Always output the best weather, light, and road type descriptions
    best_weather_last_word = best_weather.split()[-1]  # 只保留最后一个词
    results.append(f"Weather: {best_weather_last_word}\n")
    results.append(f"Lighting: {best_light}\n")
    best_road_type_last_word = best_road_type.split()[-1]  # 只保留最后一个词
    results.append(f"Road Type: {best_road_type_last_word}\n")

    # Only output road conditions and static obstacles if they exceed the threshold
    if road_conditions_confidence >= 0.3:
        results.append(f"Road Conditions: {best_road_conditions}\n")
    if static_obstacles_confidence >= 0.2:
        results.append(f"Static Obstacles: {best_static_obstacles}")

    # Collect confidences for heatmap
    confidences = {
        'weather': weather_confidence,
        'lighting': light_confidence,
        'road_type': road_type_confidence,
        'road_conditions': road_conditions_confidence,
        'static_obstacles': static_obstacles_confidence
    }

    return results, confidences
