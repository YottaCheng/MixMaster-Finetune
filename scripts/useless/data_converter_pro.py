import os
import re
import json
import random
from docx import Document
from typing import List, Dict, Tuple
from collections import OrderedDict


class Config:
    # 路径配置
    input_file = '/Volumes/Study/prj/data/raw/training_labeled_data.docx'
    output_file = '/Volumes/Study/prj/data/llama_factory/test_back.json'
    
    # 标签配置
    label_mapping = OrderedDict([
        ("高频", "high_freq"),
        ("中频", "mid_freq"),
        ("低频", "low_freq"),
        ("压缩", "compression"),
        ("声场", "soundstage"),
        ("reverb", "reverb"),
        ("音量", "volume"),
        ("效果", "effect")
    ])
    
    # 置信度生成规则（可自定义）
    confidence_rules = {
        "high_freq": (0.7, 0.95),    # 高频处理置信度范围
        "mid_freq": (0.6, 0.85),     # 中频处理范围
        "low_freq": (0.5, 0.75),
        "compression": (0.65, 0.9),
        "soundstage": (0.7, 0.95),
        "reverb": (0.75, 1.0),
        "volume": (0.8, 1.0),
        "effect": (0.5, 0.8)
    }
    
    # 系统提示模板
    system_prompt = """您是一个专业音乐制作助手，请根据用户对音频效果的描述，从以下标签中选择合适的处理方式..."""
    
    # 随机配置
    random_seed = 42
    test_ratio = 1.0

class DataConverter:
    def __init__(self):
        self.counter = 0
        random.seed(Config.random_seed)
        
    def _generate_confidence(self, label: str) -> float:
        """生成标签置信度"""
        min_val, max_val = Config.confidence_rules.get(label, (0.5, 1.0))
        return round(random.uniform(min_val, max_val), 2)
        
    def process_line(self, line: str) -> Dict:
        """处理单行数据（含元数据生成）"""
        self.counter += 1
        
        # 提取基础信息
        parts = re.split(r'[（）]', line)
        if len(parts) < 2:
            return None
            
        text = re.sub(r'^\d+\.\s*', '', parts[0]).strip()
        raw_labels = [lbl.strip() for lbl in re.split(r'[，,、]', parts[1])]
        
        # 标签处理
        label_confidences = {}
        for lbl in raw_labels:
            if mapped := Config.label_mapping.get(lbl):
                confidence = self._generate_confidence(mapped)
                label_confidences[mapped] = confidence
                
        if not label_confidences:
            return None
            
        # 构建metadata
        metadata = {
            "text_length": len(text),
            "label_confidence": label_confidences,
            "label_count": len(label_confidences)
        }
        
        return {
            "id": f"mix_{self.counter:04d}",
            "conversations": [
                {
                    "role": "user",
                    "content": f"{Config.system_prompt}\n问题：{text}"
                },
                {
                    "role": "assistant",
                    "content": ",".join(sorted(label_confidences.keys()))
                }
            ],
            "metadata": metadata
        }

    def generate_testset(self, data: List[Dict]) -> List[Dict]:
        """生成全量测试集"""
        random.shuffle(data)
        return data[:int(len(data)*Config.test_ratio)]
    def parse_docx(self, file_path: str) -> List[str]:
        """解析Word文档，返回非空段落列表"""
        doc = Document(file_path)
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                lines.append(text)
        return lines    

# =====================
# 主程序
# =====================
def main():
    # 初始化转换器
    converter = DataConverter()
    
    try:
        # 1. 读取数据
        if not os.path.exists(Config.input_file):
            raise FileNotFoundError(f"输入文件不存在: {Config.input_file}")
            
        raw_lines = converter.parse_docx(Config.input_file)
        print(f"✅ 成功读取 {len(raw_lines)} 条原始数据")

        # 2. 处理数据
        processed_data = []
        for line in raw_lines:
            if item := converter.process_line(line):
                processed_data.append(item)
        print(f"🔄 有效转换 {len(processed_data)} 条数据（过滤 {len(raw_lines)-len(processed_data)} 条无效数据）")

        # 3. 生成测试集
        test_data = converter.generate_testset(processed_data)
        
        # 4. 保存结果
        os.makedirs(os.path.dirname(Config.output_file), exist_ok=True)
        with open(Config.output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            
        print(f"💾 已生成测试集：{len(test_data)} 条 → {Config.output_file}")

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()