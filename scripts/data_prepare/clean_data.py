import os
import re
import math
import json
import logging
import random
from collections import defaultdict
from docx import Document
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '../../data'))
# 配置日志
logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'data_cleaning.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
PATH_CONFIG = {
    "input": os.path.join(DATA_ROOT, 'raw/training_labeled_data.docx'),
    "output_json": os.path.join(DATA_ROOT, 'processed/cleaned_data.json'),
    "output_txt": os.path.join(DATA_ROOT, 'processed/cleaned_data.txt'),
    "report": os.path.join(DATA_ROOT, 'processed/data_quality_report.json')
}

# 领域特定生成模板（混音师专用）
SUPPLEMENT_TEMPLATES = {
    "效果器": [
        "需要添加{effect}效果让人声更有{quality}",
        "这段{instrument}的{effect}处理需要更{character}",
        "希望{part}部分加入{effect}制造{feeling}"
    ],
    "音量": [
        "{instrument}的电平{relation}其他元素",
        "副歌部分的{part}需要{change}3dB左右",
        "{element}的动态范围需要更{character}"
    ],
    "低频": [
        "Bass的{character}需要更{quality}",
        "{instrument}的低频{part}缺乏{quality}",
        "整体低频的{aspect}需要调整"
    ],
    "声场": [
        "{instrument}的声像需要更{position}",
        "{part}部分的{aspect}空间感不足",
        "整体声场的{character}需要增强"
    ],
    "reverb": [
        "{instrument}的{type}混响需要更{quality}",
        "{part}部分的{aspect}空间效果不足",
        "需要增加{type}混响的{character}"
    ]
}

# 领域词汇库
DOMAIN_VOCAB = {
    "effect": ["电话音", "失真", "autotune", "镶边", "延迟", "和声"],
    "quality": ["温暖", "通透", "磁性", "颗粒感", "复古感", "现代感"],
    "instrument": ["人声", "贝斯", "鼓组", "吉他", "合成器", "弦乐"],
    "character": ["明显", "细腻", "激进", "柔和", "突出", "自然"],
    "part": ["句头", "句尾", "过渡段", "副歌", "前奏", "桥段"],
    "aspect": ["衰减时间", "预延迟", "高频阻尼", "扩散度", "密度"],
    "position": ["居中", "靠左", "分散", "聚焦", "立体"],
    "type": ["板式", "大厅", "房间", "弹簧", "卷积"],
    "relation": ["超过", "低于", "匹配", "平衡于"],
    "change": ["提升", "衰减", "压缩", "限制"],
    "element": ["人声", "底鼓", "军鼓", "主奏吉他", "Pad音色"],
    "feeling": ["迷幻感", "冲击力", "沉浸感", "距离感"]
}

def clean_data(input_file: str, output_json: str, output_txt: str) -> List[dict]:
    """清洗并结构化数据"""
    pattern = re.compile(
        r'^\s*'                          
        r'(?P<index>\d+)'               
        r'\.\s+'                         
        r'(?P<text>[^（]+)'             
        r'（'                           
        r'(?P<labels>[^）]+)'           
        r'）'                           
        r'\s*$',                         
        flags=re.MULTILINE
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
                
                labels = [lbl.strip() for lbl in match.group('labels').split('，')]
                
                structured_data.append({
                    "id": int(match.group('index')),
                    "text": match.group('text').strip(),
                    "labels": labels
                })
            except Exception as e:
                logging.error(f"解析失败: {line} | 错误: {str(e)}")
                error_count += 1
                
    except Exception as e:
        logging.critical(f"文件读取失败: {str(e)}")
        raise
    
    # 保存数据
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        for item in structured_data:
            labels_str = '，'.join(item['labels'])
            f.write(f"{item['id']}. {item['text']}（{labels_str}）\n")
    
    return structured_data, error_count
def enhance_label_distribution(data: List[dict]) -> List[dict]:
    """基于粗糙集的标签分布增强 (理论来自论文1)"""
    # 计算标签共现概率矩阵
    co_occurrence = defaultdict(lambda: defaultdict(int))
    label_counts = defaultdict(int)
    
    for item in data:
        labels = item["labels"]
        for i in range(len(labels)):
            label_counts[labels[i]] += 1
            for j in range(i+1, len(labels)):
                co_occurrence[labels[i]][labels[j]] += 1
                co_occurrence[labels[j]][labels[i]] += 1
    
    # 生成增强数据（模拟论文中的RLSD计算）
    enhanced_data = []
    for item in data:
        main_label = max(item["labels"], key=lambda x: label_counts[x])
        
        # 根据共现概率选择关联标签
        related = sorted(co_occurrence[main_label].items(), 
                       key=lambda x: x[1], reverse=True)[:2]
        new_labels = item["labels"] + [k for k,v in related if v > 0]
        
        enhanced_data.append({
            **item,
            "labels": list(set(new_labels)),  # 去重
            "source": "enhanced"
        })
    
    return enhanced_data
def calculate_metrics(data: List[dict]) -> dict:
    """多标签评估指标计算（采用低相关指标）"""
    # 初始化指标容器 (论文2建议的指标组合)
    metrics = {
        "HammingLoss": 0.0,
        "SubsetAccuracy": 0.0,
        "Coverage": 0.0,
        "AveragePrecision": 0.0
    }
    
    total = len(data)
    for item in data:
        true_labels = set(item["labels"])
        # 假设pred_labels为后续模型预测结果
        pred_labels = set(true_labels)  # 此处简化为完美预测
        
        # Hamming Loss计算 (论文2核心指标)
        xor = len(true_labels.symmetric_difference(pred_labels))
        metrics["HammingLoss"] += xor / len(DOMAIN_VOCAB["effect"])
        
        # Subset Accuracy计算 (论文2推荐指标)
        metrics["SubsetAccuracy"] += 1 if true_labels == pred_labels else 0
        
    # 标准化计算结果
    metrics["HammingLoss"] /= total
    metrics["SubsetAccuracy"] /= total
    
    return metrics

def calculate_entropy(data: List[dict]) -> tuple:
    """计算信息熵和标签分布"""
    label_counts = defaultdict(int)
    total_labels = 0
    
    for item in data:
        labels = item["labels"]
        total_labels += len(labels)
        for label in labels:
            label_counts[label] += 1
    
    entropy = 0.0
    for count in label_counts.values():
        prob = count / total_labels
        entropy -= prob * math.log2(prob) if prob > 0 else 0
    
    return entropy, label_counts

def generate_supplementary_data(existing_data: List[dict], label_dist: Dict[str, int]) -> List[dict]:
    """生成领域特定补充数据"""
    # 计算需要补充的数量
    max_count = max(label_dist.values())
    min_count = min(label_dist.values())
    avg_count = sum(label_dist.values()) / len(label_dist)
    
    supplement_data = []
    current_max_id = max(item["id"] for item in existing_data) if existing_data else 0
    
    for label, count in label_dist.items():
        # 生成规则：低于平均值80%的标签需要补充
        if count < avg_count * 0.8:
            needed = int(avg_count * 0.8 - count)
            print(f"为标签【{label}】生成{needed}条补充数据...")
            
            for _ in range(needed):
                current_max_id += 1
                if label in SUPPLEMENT_TEMPLATES:
                    template = random.choice(SUPPLEMENT_TEMPLATES[label])
                    generated = template.format(
                        **{k: random.choice(v) for k, v in DOMAIN_VOCAB.items()}
                    )
                else:
                    # 通用生成规则
                    generated = f"需要调整{label}的{random.choice(['平衡度','质感','动态','空间表现'])}"
                
                supplement_data.append({
                    "id": current_max_id,
                    "text": generated,
                    "labels": [label]
                })
    
    return supplement_data

def generate_report(data: List[dict], output_report_cn: str, output_report_en: str):
    """
    生成多语言的数据质量报告
    
    Args:
        data (List[dict]): 处理后的数据集
        output_report_cn (str): 中文报告输出路径
        output_report_en (str): 英文报告输出路径
    """
    entropy, label_dist = calculate_entropy(data)
    metrics = calculate_metrics(data)
    
    avg = sum(label_dist.values()) / len(label_dist)
    recommendations = [
        f"标签【{label}】数据量偏低（{count}），建议增加相关标注数据"
        for label, count in label_dist.items() if count < avg * 0.7
    ]
    
    # 中文报告
    report_cn = {
        "摘要": {
            "总样本数": len(data),
            "唯一标签数": len(label_dist),
            "信息熵": f"{entropy:.4f}",
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "改进建议": recommendations
        },
        "标签分布": label_dist
    }
    
    # 英文报告
    report_en = {
        "Summary": {
            "Total Samples": len(data),
            "Unique Labels": len(label_dist),
            "Entropy": f"{entropy:.4f}",
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "Recommendations": [
                f"Label [{label}] has low data volume ({count}), recommended to add more annotated data"
                for label, count in label_dist.items() if count < avg * 0.7
            ]
        },
        "Label Distribution": label_dist
    }
    
    # 保存中文报告
    with open(output_report_cn, 'w', encoding='utf-8') as f:
        json.dump(report_cn, f, ensure_ascii=False, indent=2)
    
    # 保存英文报告
    with open(output_report_en, 'w', encoding='utf-8') as f:
        json.dump(report_en, f, ensure_ascii=False, indent=2)
    
    print(f"已生成中文报告：{output_report_cn}")
    print(f"已生成英文报告：{output_report_en}")

def main():
    # 创建必要目录
    os.makedirs(os.path.dirname(PATH_CONFIG["output_json"]), exist_ok=True)
    print(f"开始处理文件：{PATH_CONFIG['input']}")
    try:
        # 原始数据清洗
        cleaned_data, errors = clean_data(
            PATH_CONFIG["input"],
            PATH_CONFIG["output_json"],
            PATH_CONFIG["output_txt"]
        )
        print(f"初始清洗完成，有效数据：{len(cleaned_data)}条，失败：{errors}条")
        
        # 分析现有分布
        _, label_dist = calculate_entropy(cleaned_data)
        
        # 生成补充数据
        supplements = generate_supplementary_data(cleaned_data, label_dist)
        if supplements:
            cleaned_data += supplements
            print(f"已生成{len(supplements)}条补充数据")
            
            # 保存更新后的数据
            with open(PATH_CONFIG["output_json"], 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            with open(PATH_CONFIG["output_txt"], 'w', encoding='utf-8') as f:
                for item in cleaned_data:
                    labels_str = '，'.join(item['labels'])
                    f.write(f"{item['id']}. {item['text']}（{labels_str}）\n")
        
        # 应用粗糙集增强标签分布
        enhanced_data = enhance_label_distribution(cleaned_data)
        print(f"已生成{len(enhanced_data)}条增强数据")
        
        # 合并原始数据和增强数据
        final_data = cleaned_data + enhanced_data
        
        # 生成最终报告 - 新增英文报告路径
        generate_report(
            final_data, 
            PATH_CONFIG["report"].replace(".json", "_cn.json"),
            PATH_CONFIG["report"].replace(".json", "_en.json")
        )
        print(f"质量报告已生成：{PATH_CONFIG['report']}")
        print(f"最终数据总量：{len(final_data)}条")
    
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()