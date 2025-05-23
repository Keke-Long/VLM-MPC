
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt


# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("huggingface/llama3")
tokenizer = AutoTokenizer.from_pretrained("huggingface/llama3")

# 生成输入
prompt = read_txt_file("/home/ubuntu/Documents/VLM_MPC/scenes_data/0ac05652a4c44374998be876ba5cd6fd/b_LLM_MPC/prompt.txt")
inputs = tokenizer(prompt, return_tensors="pt")

# 获取输出和注意力矩阵
outputs = model(**inputs, output_attentions=True)
attention_matrices = outputs.attentions  # 这是注意力矩阵

# 可视化或分析注意力矩阵
