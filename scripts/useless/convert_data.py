import os
import json
import re

# 输入和输出路径
input_file = r"D:\kings\MixMaster-Finetune\data\processed\cleaned_data.txt"
output_dir = r"D:\kings\MixMaster-Finetune\LLaMA-Factory\data\mix"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取文本文件
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 解析数据
data = []
for line in lines:
    # 去除行号和多余空格
    line = line.strip()
    if not line:
        continue

    # 使用正则表达式提取文本内容和标签
    match = re.match(r"^(.*?)（(.*?)）$", line)
    if not match:
        print(f"无法解析行：{line}")
        continue

    text = match.group(1).strip()  # 提取文本内容
    label = match.group(2).strip()  # 提取标签
    data.append({"text": text, "label": label})

# 转换为指令微调格式
instruction_data_path = os.path.join(output_dir, "instruction_data.jsonl")
with open(instruction_data_path, "w", encoding="utf-8") as f:
    for item in data:
        formatted_item = {
            "instruction": "请对以下文本进行分类：",
            "input": item["text"],
            "output": item["label"]
        }
        f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")

print(f"指令微调格式数据已保存到：{instruction_data_path}")

# 转换为文本分类格式
classification_data_path = os.path.join(output_dir, "classification_data.txt")
with open(classification_data_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(f"{item['text']}\t{item['label']}\n")

print(f"文本分类格式数据已保存到：{classification_data_path}")