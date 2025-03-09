import os
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------- 翻译模块依赖处理 ----------
try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("\n❌ 缺少必要依赖包：deep-translator")
    print("请执行以下命令安装：")
    print("pip install deep-translator")
    print("或使用清华镜像源加速安装：")
    print("pip install deep-translator -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

# ---------- 混音标签预测器 ----------
class MixingLabelPredictor:
    def __init__(self, model_dir=r"D:\kings\prj\MixMaster-Finetune\config\Models\deepseek_R1_MixMaster"):
        # 验证路径
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ 模型目录不存在: {model_dir}")
        
        # 加载模型组件
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 加载标签映射
        self.label_mapping = self._load_label_mapping(
            os.path.join(model_dir, "music_synonyms.json")
        
        # 初始化翻译器
        self.translator = GoogleTranslator(source='zh-CN', target='en')
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _load_label_mapping(self, path):
        """加载标签映射文件"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _contains_chinese(self, text):
        """中文检测"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def predict(self, input_text, lang="中文"):
        """执行预测"""
        try:
            # 翻译处理
            if self._contains_chinese(input_text):
                translated_text = self.translator.translate(input_text)
            else:
                translated_text = input_text

            # 文本编码
            inputs = self.tokenizer(
                translated_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取预测结果
            predicted_idx = torch.argmax(outputs.logits, dim=1).item()
            return (
                self.label_mapping[str(predicted_idx)]["zh"],
                self.label_mapping[str(predicted_idx)]["en"],
                str(predicted_idx)
        except Exception as e:
            return f"❌ 预测失败", f"❌ 错误代码", str(e)

# ---------- 测试运行 ----------
if __name__ == "__main__":
    try:
        predictor = MixingLabelPredictor()
        test_text = "人声高频需要更明亮"
        zh_label, en_label, code = predictor.predict(test_text)
        print("\n" + "="*40)
        print(f"📥 输入：{test_text}")
        print(f"🇨🇳 中文标签：{zh_label}")
        print(f"🇺🇸 英文标签：{en_label}")
        print(f"🔢 标签代码：{code}")
        print("="*40)
    except Exception as e:
        print(f"\n❌ 初始化失败：{str(e)}")
        print("可能原因：")
        print("1. 模型文件缺失或不完整")
        print("2. 标签映射文件格式错误")
        print("3. 未安装必要依赖（torch/transformers）")
        print("解决方案：")
        print("pip install torch transformers")