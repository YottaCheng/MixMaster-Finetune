import json
import os
import re
from docx import Document
from collections import defaultdict

VALID_LABELS = {'低频', '中频', '高频', 'reverb', '效果器', '声场', '压缩', '音量'}
MAX_LABELS = 3

def clean_text(text):
    """改进的文本清洗函数"""
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s，。？！、：；《》（）“”‘’]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate_diverse_instructions():  # <--- 新增缺失的函数
    """基于Self-Instruct思想的多样化指令生成"""
    base_instructions = [
        "分析音频处理需求并选择相关参数（1-3个）：选项：低频/中频/高频/reverb/效果器/声场/压缩/音量",
        #"识别需要调整的关键参数：",
        #"根据描述确定处理方案："
    ]
    variants = ["请选择", "请识别"]
    instructions = []
    for base in base_instructions:
        for var in variants:
            instructions.append(f"{var}{base}")
            instructions.append(f"{base}（最多选3项）")
    return list(set(instructions))[:10]

def extract_labels(text):
    """智能标签提取函数"""
    bracket_labels = re.findall(r'[（(]([^）)]+)[）)]', text)
    if bracket_labels:
        return [lb.strip() for lb in re.split(r'[，,]\s*', ''.join(bracket_labels))]
    
    if ':' in text or '：' in text:
        sep = ':' if ':' in text else '：'
        _, labels_part = text.split(sep, 1)
        return [lb.strip() for lb in re.split(r'[，,]\s*', labels_part)]
    
    return []

def parse_docx(file_path):
    """增强版文档解析函数（支持多行数据）"""
    doc = Document(file_path)
    data = []
    
    for para in doc.paragraphs:
        # 按换行符分割多行数据
        lines = [line.strip() for line in para.text.split('\n') if line.strip()]
        
        for raw_text in lines:  # 逐行处理
            if not raw_text:
                continue

            # 智能标签提取（新增多标签分割）
            labels = extract_labels(raw_text)
            valid_labels = []
            for lb in labels:
                # 新增多标签分割（支持中文顿号、斜杠）
                sub_labels = re.split(r'[，,、/]', lb)
                valid_labels.extend([s.strip() for s in sub_labels if s.strip() in VALID_LABELS])
            
            valid_labels = list(set(valid_labels))[:MAX_LABELS]  # 去重
            
            # 获取输入文本（优化编号去除）
            input_text = re.sub(r'^\d+[\.、]?\s*', '', raw_text)  # 支持"81."、"81、"等格式
            input_text = re.sub(r'[（(].*?[）)]', '', input_text)  # 移除括号内容
            input_text = re.sub(r':.*', '', input_text).strip()
            input_text = clean_text(input_text)
            
            if not valid_labels:
                print(f"⚠️ 无效标签：{labels} | 原始文本：'{raw_text}'")
                continue
                
            data.append({
                "text": input_text,
                "labels": valid_labels
            })
    
    return data

def balance_check(labels_dist):
    """鲁棒的平衡检测函数"""
    if not labels_dist:
        print("⚠️ 未检测到有效标签，跳过平衡检查")
        return {}
    
    total = sum(labels_dist.values())
    try:
        avg = total / len(labels_dist)
        return {lb: cnt/avg for lb, cnt in labels_dist.items() if cnt/avg > 2 or cnt/avg < 0.5}
    except ZeroDivisionError:
        print("⚠️ 标签统计异常，请检查数据格式")
        return {}

def convert_test_data():
    input_path = "/Volumes/Study/prj/data/raw/test_raw_data.docx"
    output_path = "/Volumes/Study/prj/data/llama_factory/test.json"
    
    try:
        raw_data = parse_docx(input_path)
    except Exception as e:
        print(f"文档解析失败：{str(e)}")
        return

    alpaca_data = []
    labels_dist = defaultdict(int)
    
    for item in raw_data:
        valid_labels = item.get("labels", [])
        for lb in valid_labels:
            labels_dist[lb] += 1
        
        # 使用定义好的指令生成函数
        for instr in generate_diverse_instructions():  # <--- 现在函数已定义
            alpaca_data.append({
                "instruction": instr,
                "input": item["text"],
                "output": ", ".join(valid_labels) if valid_labels else "无相关需求"
            })
    
    if imbalance := balance_check(labels_dist):
        print(f"警告：标签不平衡 {imbalance}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成完成！样本数：{len(alpaca_data)}\n标签分布：{dict(labels_dist)}")

if __name__ == "__main__":
    convert_test_data()