import json
import os
import re

VALID_LABELS = {'低频', '中频', '高频', 'reverb', '效果器', '声场', '压缩', '音量'}

def clean_text(text):
    # Remove numbers and punctuation simultaneously
    text = re.sub(r'[\d_]', '', text)  # New addition to filter out numbers and underscores
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text)  # Remove punctuation
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
    # 使用 os.path.dirname(__file__) 获取当前脚本的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建绝对路径 - 注意根据你的实际项目结构调整
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # 返回项目根目录
    input_path = os.path.join(base_dir, "data", "processed", "modified_data.json")
    output_dir = os.path.join(base_dir, "data", "llama_factory")
    
    # 打印路径以便调试
    print(f"Input path: {input_path}")
    print(f"Output path: {output_dir}")
    
    # 确保路径存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.json")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 后续代码保持不变
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