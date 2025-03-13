import json

# 文件路径
file_path = r"D:\kings\prj\Finetune_local\LLaMA-Factory\data\mix\test.json"

# 读取原始数据
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 为每个样本添加空的 input 字段
for item in data:
    item["input"] = ""  # 设置 input 为空字符串

# 保存修改后的数据
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"成功为 {len(data)} 条数据添加空的 input 字段！")