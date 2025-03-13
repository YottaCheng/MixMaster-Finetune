import json
import os
import re

VALID_LABELS = {'低频', '中频', '高频', 'reverb', '效果器', '声场', '压缩', '音量'}

def clean_text(text):
    # 同时移除数字和标点符号
    text = re.sub(r'[\d_]', '', text)  # 新增数字和下划线过滤
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text)  # 移除0-9数字
    return re.sub(r'\s+', ' ', text).strip()

def generate_instructions():
    """生成带数量限制的多样化指令"""
    return [
        # 核心指令（明确数量限制）
        "分类音频处理需求（选择1-3个相关参数）：选项：低频/中频/高频/reverb/效果器/声场/压缩/音量",
        
        # 新增变体指令（保持多样性）
        "识别需要调整的1-3个关键音频参数：",
        "提取主要音效处理类型（最多三项）：",
        "从以下需求中选择至多三个处理需求：",
        "分析需要优化的音频参数（选择最重要的一到三个）："
    ]

def convert_to_alpaca():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, "../data/processed/cleaned_data.json"))
    
    # 修改点1：output_dir应该指向目录路径
    output_dir = os.path.normpath(os.path.join(script_dir, "../data/llama_factory"))
    # 修改点2：输出文件路径单独定义
    output_path = os.path.join(output_dir, "train.json")
    
    os.makedirs(output_dir, exist_ok=True)  # 现在正确创建目录
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    alpaca_data = []
    for item in raw_data:
        if not item.get("text") or not item.get("labels"):
            continue
        
        cleaned_input = clean_text(item["text"])
        valid_labels = [lb for lb in item["labels"] if lb in VALID_LABELS][:3]

        for instr in generate_instructions():
            alpaca_data.append({
                "instruction": instr,
                "input": cleaned_input,
                "output": ", ".join(valid_labels) if valid_labels else "无相关需求"
            })
    
    # 使用正确的输出路径
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成完成！样本数：{len(alpaca_data)} | 路径：{output_path}")
if __name__ == "__main__":
    convert_to_alpaca()