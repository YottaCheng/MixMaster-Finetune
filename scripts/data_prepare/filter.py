import re
import torch
import numpy as np
from docx import Document
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置参数
MIN_SIMILARITY = 0.75  # 语义相似度阈值
AUDIO_TERMS = {
    "混响", "声场", "人声", "伴奏", "高频", "低频", 
    "压缩", "EQ", "颗粒感", "空间感", "立体声",
    "Autotune", "中频", "鼻音", "工业", "音乐厅"
}

def load_docx(file_path):
    """加载Word文档内容（论文1/3方法）"""
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        return []
    logging.info(f"加载文档: {file_path}")
    doc = Document(file_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    logging.info(f"加载完成，共{len(paragraphs)}段落")
    return paragraphs

def clean_data(texts):
    """数据清洗：去重/去噪/标准化（论文1/3/6方法）"""
    logging.info("开始数据清洗")
    cleaned = []
    seen = set()
    for text in texts:
        normalized = re.sub(r'\s+', ' ', text).strip()
        # 新增：按冒号分割处理
        if "：" in normalized:
            key = normalized.split("：", 1)[1]
        else:
            key = normalized
        if key and key not in seen:
            seen.add(key)
            cleaned.append(normalized)
    logging.info(f"清洗前{len(texts)}条，清洗后{len(cleaned)}条")
    return cleaned

def bert_similarity(text1, text2, tokenizer, model):
    """基于BERT的语义相似度计算（论文4/5/8方法）"""
    inputs = tokenizer([text1, text2], return_tensors='pt', 
                      padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return cosine_similarity(embeddings[0].unsqueeze(0), 
                            embeddings[1].unsqueeze(0)).item()

def filter_variants(original, variants, tokenizer, model):
    """过滤变体数据（EDA/回译/反义）"""
    logging.debug(f"开始过滤变体（原始：{original}）")
    filtered = []
    seen = set()  # 新增：用于存储已存在的变体内容
    for var in variants:
        # 提取冒号后内容作为唯一键
        var_key = var.split("：", 1)[1] if "：" in var else var
        original_key = original.split("：", 1)[1] if "：" in original else original
        
        # 新增过滤条件：内容重复或与原始重复
        if var_key == original_key or var_key in seen:
            logging.debug(f"直接过滤重复变体: {var}")
            continue
        
        seen.add(var_key)
        sim = bert_similarity(original, var, tokenizer, model)
        if sim >= MIN_SIMILARITY:
            filtered.append(var)
            logging.debug(f"保留变体: {var} (相似度: {sim:.4f})")
        else:
            logging.debug(f"过滤变体: {var} (相似度: {sim:.4f})")
    return filtered

def domain_filter(texts):
    """领域术语过滤（论文2/7方法）"""
    filtered = [t for t in texts if any(term in t for term in AUDIO_TERMS)]
    logging.info(f"领域过滤前{len(texts)}条，过滤后{len(filtered)}条")
    return filtered

def process_entry(entry, tokenizer, model):
    """处理单条数据记录"""
    logging.debug(f"处理条目: {entry[:50]}...")
    parts = re.split(r'\n+', entry)
    original = parts[0].split("：", 1)[1].strip()
    variants = [p.split("：", 1)[1].strip() for p in parts[1:] 
               if p.startswith(("EDA变体", "回译结果", "反义结果"))]
    
    logging.debug(f"原始数据: {original}")
    logging.debug(f"原始变体数量: {len(variants)}")
    
    # 新增：对原始数据和变体进行内容去重
    filtered_vars = []
    seen = set()
    for var in variants:
        var_key = var.split("：", 1)[1] if "：" in var else var
        original_key = original.split("：", 1)[1] if "：" in original else original
        
        if var_key != original_key and var_key not in seen:
            seen.add(var_key)
            filtered_vars.append(var)
    
    # 再进行BERT过滤
    bert_filtered = filter_variants(original, filtered_vars, tokenizer, model)
    
    results = [f"原始数据：{original}"]
    results.extend([f"有效变体：{var}" for var in bert_filtered])
    logging.debug(f"过滤后变体数量: {len(bert_filtered)}")
    return results

def main():
    logging.info("初始化BERT模型")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    
    logging.info("加载数据文件")
    augmented = load_docx("/Volumes/Study/prj/data/processed/augmented_results.docx")
    backtrans = load_docx("/Volumes/Study/prj/data/processed/backtrans_results.docx")
    
    logging.info("数据清洗")
    cleaned_aug = clean_data(augmented)
    cleaned_back = clean_data(backtrans)
    
    logging.info("合并数据集")
    merged_data = cleaned_aug + cleaned_back
    logging.info(f"合并后总条目数: {len(merged_data)}")
    
    # 再次去重
    logging.info("再次去重")
    merged_data = clean_data(merged_data)
    logging.info(f"再次去重后总条目数: {len(merged_data)}")
    
    # 以下步骤被注释，保留到再次去重后的数据
    # logging.info("处理数据条目")
    # final_data = []
    # for entry in merged_data:
    #     if entry.startswith("原始数据："):
    #         try:
    #             processed = process_entry(entry, tokenizer, model)
    #             final_data.extend(processed)
    #         except Exception as e:
    #             logging.error(f"处理条目失败: {entry[:50]}...", exc_info=True)
    
    # logging.info("领域过滤")
    # final_data = domain_filter(final_data)
    
    # 直接使用去重后的数据
    final_data = merged_data
    
    logging.info("保存结果文件")
    doc = Document()
    for idx, line in enumerate(final_data, 1):
        doc.add_paragraph(f"{idx}. {line}")
    doc.save("/Volumes/Study/prj/data/processed/filtered_results.docx")
    logging.info(f"保存完成，共{len(final_data)}条记录")

if __name__ == "__main__":
    main()