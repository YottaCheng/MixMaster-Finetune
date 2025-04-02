import json
import os
import re
from docx import Document
from collections import defaultdict

VALID_LABELS = {'低频', '中频', '高频', 'reverb', '效果器', '声场', '压缩', '音量'}
MAX_LABELS = 3

# 设置相对路径（相对于scripts目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODIFIED_DATA_PATH = os.path.join(BASE_DIR, "..","data", "llama_factory", "modified_data.json")
DOCX_INPUT_PATH = os.path.join(BASE_DIR, "..","data", "raw", "test_raw_data.docx")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "..","data", "llama_factory", "alpaca_data.json")

def clean_text(text):
    """改进的文本清洗函数"""
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s，。？！、：；《》（）""'']', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate_diverse_instructions():
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

def convert_json_to_alpaca(input_path, output_path=None):
    """将JSON文件从id-text-labels格式转换为instruction-input-output格式"""
    # 设置默认输出路径
    if output_path is None:
        base_dir = os.path.dirname(input_path)
        output_path = os.path.join(base_dir, "alpaca_data.json")
    
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式
        if not isinstance(data, list):
            print(f"❌ 数据格式错误：预期列表，实际为 {type(data)}")
            return None
        
        # 转换数据
        alpaca_data = []
        labels_dist = defaultdict(int)
        
        # 检查文件格式
        is_id_text_format = all(isinstance(item, dict) and 'id' in item and 'text' in item and 'labels' in item 
                               for item in data[:5]) if len(data) >= 5 else False
        
        if is_id_text_format:
            print("✓ 检测到id-text-labels格式，进行转换...")
            for item in data:
                valid_labels = item.get("labels", [])
                text = item.get("text", "")
                
                # 统计标签分布
                for lb in valid_labels:
                    labels_dist[lb] += 1
                
                # 生成多样化指令并添加到数据中
                for instr in generate_diverse_instructions():
                    alpaca_data.append({
                        "instruction": instr,
                        "input": text,
                        "output": ", ".join(valid_labels) if valid_labels else "无相关需求"
                    })
        else:
            print("✓ 保持原始格式...")
            return None
        
        # 检查标签分布平衡性
        if imbalance := balance_check(labels_dist):
            print(f"⚠️ 警告：标签不平衡 {imbalance}")
        
        # 保存转换后的数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 生成完成！样本数：{len(alpaca_data)}\n标签分布：{dict(labels_dist)}")
        return output_path
    
    except Exception as e:
        print(f"❌ 转换失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
    """主函数，处理测试数据"""
    print(f"当前脚本路径: {os.path.abspath(__file__)}")
    print(f"基础目录: {BASE_DIR}")
    print(f"检查文件: {MODIFIED_DATA_PATH}")
    
    # 策略1：如果检测到modified_data.json，执行转换
    if os.path.exists(MODIFIED_DATA_PATH):
        print(f"✓ 检测到文件 {os.path.relpath(MODIFIED_DATA_PATH, BASE_DIR)}，执行JSON转换...")
        output_file = convert_json_to_alpaca(MODIFIED_DATA_PATH, DEFAULT_OUTPUT_PATH)
        if output_file:
            print(f"✅ 转换成功! 结果已保存到: {os.path.relpath(output_file, BASE_DIR)}")
        else:
            print("❌ 转换失败!")
        return
    
    # 策略2：如果没有检测到modified_data.json，保持原代码逻辑
    print(f"✓ 未检测到文件 {os.path.relpath(MODIFIED_DATA_PATH, BASE_DIR)}，执行原始DOCX转换流程...")
    try:
        # 确保输入文件存在
        if not os.path.exists(DOCX_INPUT_PATH):
            print(f"❌ 输入文件不存在: {os.path.relpath(DOCX_INPUT_PATH, BASE_DIR)}")
            print(f"请确保文件位于正确位置，或修改脚本中的路径配置。")
            return
            
        raw_data = parse_docx(DOCX_INPUT_PATH)
    except Exception as e:
        print(f"❌ 文档解析失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return

    alpaca_data = []
    labels_dist = defaultdict(int)
    
    for item in raw_data:
        valid_labels = item.get("labels", [])
        for lb in valid_labels:
            labels_dist[lb] += 1
        
        # 使用指令生成函数
        for instr in generate_diverse_instructions():
            alpaca_data.append({
                "instruction": instr,
                "input": item["text"],
                "output": ", ".join(valid_labels) if valid_labels else "无相关需求"
            })
    
    if imbalance := balance_check(labels_dist):
        print(f"⚠️ 警告：标签不平衡 {imbalance}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
    with open(DEFAULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成完成！")
    print(f"✅ 样本数：{len(alpaca_data)}")
    print(f"✅ 标签分布：{dict(labels_dist)}")
    print(f"✅ 结果已保存到: {os.path.relpath(DEFAULT_OUTPUT_PATH, BASE_DIR)}")

if __name__ == "__main__":
    convert_test_data()