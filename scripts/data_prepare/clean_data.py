import re
import math
import json
import logging
from collections import defaultdict
from docx import Document

# 配置日志
logging.basicConfig(
    filename='data_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_data(input_file, output_json, output_txt):
    """清洗并结构化数据"""
    pattern = re.compile(
        r'^(?P<index>\d+)\.\s+.*?：\s*'
        r'(?P<text>.+?)'
        r'\s*（(?P<label>.+?)）\s*$'  # 修改字段名
    )
    
    structured_data = []
    error_count = 0
    
    try:
        doc = Document(input_file)
        for para in doc.paragraphs:
            line = para.text.strip()
            if not line:
                continue
            
            try:
                match = pattern.match(line)
                if not match:
                    raise ValueError("格式不匹配")
                
                groups = match.groupdict()
                
                # 修改字段结构
                structured_data.append({
                    "id": int(groups['index']),
                    "text": groups['text'].strip(),
                    "label": groups['label'].strip()  # 直接存储字符串
                })
            except Exception as e:
                logging.error(f"解析失败: {line} | 错误: {str(e)}")
                error_count += 1
                
    except Exception as e:
        logging.critical(f"文件读取失败: {str(e)}")
        raise
    
    # 保存结构化数据为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    # 保存清洗后的数据为 TXT
    with open(output_txt, 'w', encoding='utf-8') as f:
        for item in structured_data:
            f.write(f"{item['id']}. {item['text']}（{item['label']}）\n")  # 修改输出格式
    
    return structured_data, error_count

def calculate_entropy(data):
    """计算信息熵（适配单标签）"""
    label_counts = defaultdict(int)
    total_samples = len(data)
    
    for item in data:
        label_counts[item["label"]] += 1  # 修改统计方式
    
    entropy = 0.0
    for count in label_counts.values():
        prob = count / total_samples
        entropy -= prob * math.log2(prob) if prob > 0 else 0
    
    return entropy, label_counts

def generate_report(data, output_report):
    """生成质量报告"""
    entropy, label_dist = calculate_entropy(data)
    
    report = {
        "summary": {
            "total_samples": len(data),
            "unique_labels": len(label_dist),
            "entropy": f"{entropy:.4f}"
        },
        "label_distribution": label_dist
    }
    
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    input_file = '/Volumes/Study/prj/data/raw/training_labeled_data.docx'
    output_json = '/Volumes/Study/prj/data/processed/cleaned_data.json'
    output_txt = '/Volumes/Study/prj/data/processed/cleaned_data.txt'
    report_file = '/Volumes/Study/prj/data/processed/data_quality_report.json'
    
    print(f"开始处理文件：{input_file}")
    try:
        cleaned_data, errors = clean_data(input_file, output_json, output_txt)
        print(f"清洗完成，有效数据：{len(cleaned_data)}条，失败：{errors}条")
        
        generate_report(cleaned_data, report_file)
        print(f"质量报告已生成：{report_file}")
        print(f"清洗后的TXT数据已保存至：{output_txt}")
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()