import os
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator

class MixingLabelPredictor:
    def __init__(self, model_dir=r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v3"):
        # 初始化检查
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ 模型目录不存在: {model_dir}")

        # ==== 多标签关键配置 ====
        self.label_order = [
            "low_freq",    # 低频
            "mid_freq",     # 中频
            "high_freq",    # 高频
            "reverb",      # 混响
            "effect",      # 效果器
            "soundstage",   # 声场
            "compression",  # 压缩
            "volume"        # 音量
        ]
        
        # 加载标签映射
        self.label_mapping = self._load_label_mapping(
            os.path.join(model_dir, "music_master.json"))
        
        # 初始化模型（确保模型是多标签分类）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=len(self.label_order),
            problem_type="multi_label_classification",  # 关键参数
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.translator = GoogleTranslator(source='zh-CN', target='en')

    def _load_label_mapping(self, path):
        """加载与训练一致的标签映射"""
        with open(path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
        
        # 验证标签顺序与模型输出层一致
        required_labels = self.label_order
        label_mapping = {}
        for label in required_labels:
            if label not in raw_mapping:
                raise KeyError(f"❌ 缺失必需标签: {label}")
            label_mapping[label] = {
                "zh": raw_mapping[label][0],
                "en": raw_mapping[label][1]
            }
        return label_mapping

    def _contains_chinese(self, text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)


    def predict(self, input_text, lang="中文", primary_threshold=0.7, secondary_threshold=0.5):
        """优化后的预测逻辑"""
        try:
            # ==== 增强型提示词 ====
            instruction = (
                "作为专业混音师，请严格分析音频处理需求，按以下规则输出标签：\n"
                "1. 主标签（必须存在且唯一，最高置信度）\n"
                "2. 仅当明确存在其他需求时才输出副标签（1-2个）\n"
                "可用标签：低频/中频/高频/reverb/效果器/声场/压缩/音量"
            )
            
            # ==== 预处理 ====
            translated_text = self._translate_text(input_text)
            full_prompt = f"{instruction}\n输入：{translated_text}"

            # ==== 模型推理 ====
            inputs = self.tokenizer(
                full_prompt,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.sigmoid(outputs.logits).squeeze()

            # ==== 动态阈值逻辑 ====
            sorted_indices = torch.argsort(probs, descending=True)
            
            # 主标签必须超过阈值
            main_label = None
            if probs[sorted_indices[0]] >= primary_threshold:
                main_label = self.label_order[sorted_indices[0]]
            else:
                main_label = self.label_order[sorted_indices[0]]
            
            # 副标签筛选逻辑
            secondary_labels = []
            main_prob = probs[sorted_indices[0]]
            for idx in sorted_indices[1:]:
                if (probs[idx] >= secondary_threshold and 
                    (main_prob - probs[idx]) < 0.3 and 
                    len(secondary_labels) < 1):
                    label = self.label_order[idx]
                    secondary_labels.append(label)
            
            # ==== 最终标签处理 ====
            all_labels = [main_label] + secondary_labels
            all_labels = list(dict.fromkeys(all_labels))  # 去重
            
            # 语言处理
            zh_labels = [self.label_mapping[l]["zh"] for l in all_labels]
            en_labels = [self.label_mapping[l]["en"] for l in all_labels]
            
            # 返回格式
            if lang == "中文":
                return (
                    zh_labels[0] if zh_labels else "",
                    "，".join(zh_labels[1:]) if len(zh_labels)>1 else "",
                    ",".join(all_labels)
                )
            else:
                return (
                    en_labels[0] if en_labels else "",
                    ", ".join(en_labels[1:]) if len(en_labels)>1 else "",  # 英文用逗号+空格
                    ",".join(all_labels)
                )
            
        except Exception as e:
            return "❌ 预测失败", "❌ 错误", str(e)

    def _translate_text(self, text):
        """优化翻译逻辑"""
        if self._contains_chinese(text):
            return GoogleTranslator(source='zh-CN', target='en').translate(text).lower()
        return text.lower()

# 测试案例（必须放在类外）
if __name__ == "__main__":
    try:
        predictor = MixingLabelPredictor()
        test_cases = [  # 正确缩进
            ("人声高频需要更明亮", ["高频"]),
            ("增加鼓组的空间感和压缩感", ["声场", "压缩"]),
            ("整体低频太多", ["低频"]),
            ("The vocals need more air", ["high_freq"]),
            ("降低混响量", ["reverb"])
        ]

        for text, expected in test_cases:
            zh, en, code = predictor.predict(text)
            print(f"输入：{text}")
            print(f"预测：{zh} | {code}")
            print(f"预期：{expected}")
            assert code.split(",") == expected, "测试未通过"
            print("✅ 测试通过\n")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")