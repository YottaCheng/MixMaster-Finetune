import os
import re
import csv
import json
from docx import Document

def load_label_mapping(csv_path):
    """加载中英文标签映射表"""
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn = row['中文标签'].strip()
            en = row['英文标签'].strip()
            mapping[cn] = en
    return mapping

def process_docx_to_jsonl(
    docx_path,
    output_jsonl,
    label_mapping,
    bracket_pattern=r'[(（](.*?)[)）]'
):
    """
    处理 DOCX 文件并生成 JSONL 格式的输出文件
    """
    # 加载文档
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    
    # 打开 JSONL 输出文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, para in enumerate(paragraphs, 1):
            # 提取原始标签
            match = re.search(bracket_pattern, para)
            raw_label = match.group(1).strip() if match else "Unknown"
            
            # 映射英文标签
            en_label = label_mapping.get(raw_label, "Unknown")
            
            # 清理标签特殊字符
            safe_label = re.sub(r'[\\/*?:"<>|]', '_', en_label)
            
            # 移除原文中的标签内容
            content = re.sub(bracket_pattern, '', para).strip()
            
            # 构造 JSON 对象
            entry = {
                "id": f"{idx:03d}",
                "label": safe_label,
                "text": content
            }
            
            # 写入 JSONL 文件
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")  # JSONL 每行一个对象
            
            print(f"[Processed] ID: {entry['id']} | Label: {safe_label} | Text: {content[:50]}...")

def main():
    # 输入路径配置
    csv_path = "/Volumes/Study/prj/data/raw/label_mapping.csv"
    docx_path = "/Volumes/Study/prj/data/raw/training_labeled_data.docx"
    output_jsonl = "/Volumes/Study/prj/data/processed/training_data.jsonl"
    
    try:
        # 加载标签映射
        label_map = load_label_mapping(csv_path)
        print("✅ Loaded label mapping:")
        for cn, en in label_map.items():
            print(f"  {cn} → {en}")
        
        # 处理文档并生成 JSONL 文件
        process_docx_to_jsonl(
            docx_path=docx_path,
            output_jsonl=output_jsonl,
            label_mapping=label_map
        )
        
        print(f"\n🎉 All done! Output JSONL file: {output_jsonl}")
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e.filename}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()