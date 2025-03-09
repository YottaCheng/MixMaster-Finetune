import os
import joblib

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Please install deep-translator with: pip install deep-translator")
    import sys
    sys.exit(1)


class MixingLabelPredictor:
    def __init__(self, model_dir=r"D:\kings\prj\MixMaster-Finetune\config\Models\deepseek_R1_MixMaster"):
        """
        修改点：
        1. 模型路径指向新位置
        2. 添加路径存在性验证
        3. 从文件加载标签映射
        """
        # 验证模型目录是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        # 加载模型组件
        self.model = joblib.load(os.path.join(model_dir, "baseline_model.pkl"))
        self.vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        
        # 加载标签映射
        self.label_mapping = self._load_label_mapping(os.path.join(model_dir, "label_mapping.json"))
        
        # 初始化翻译器
        self.translator = GoogleTranslator(source='zh-CN', target='en')

    def _load_label_mapping(self, mapping_path):
        """从 JSON 文件加载标签映射"""
        import json
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 以下方法保持不变
    def translate_to_english(self, text):
        """将中文翻译为英文"""
        if self._contains_chinese(text):
            return self.translator.translate(text)
        return text

    def _contains_chinese(self, text):
        """检查是否包含中文字符"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def predict(self, input_text, lang="中文"):
        """执行预测"""
        # 语言转换
        english_text = self.translate_to_english(input_text).lower()
        
        # 向量化
        text_vector = self.vectorizer.transform([english_text])
        
        # 预测
        predicted_label = self.model.predict(text_vector)[0]
        
        # 获取标签信息
        label_info = self.label_mapping[predicted_label]
        return label_info["zh"], label_info["en"], predicted_label


# 以下 main() 测试代码保持不变
def main():
    # 命令行交互测试
    predictor = MixingLabelPredictor()
    
    print("\n混音标签预测系统（输入 'exit' 退出）")
    while True:
        text = input("\n请输入混音需求描述（中英文均可）: ").strip()
        if text.lower() == 'exit':
            break
            
        predicted_label_zh, predicted_label_en, label_code = predictor.predict(text, lang="中文")
        print("\n=== 预测结果 ===")
        print(f"原始输入: {text}")
        print(f"中文标签: {predicted_label_zh}")
        print(f"英文标签: {predicted_label_en}")
        print(f"标签代码: {label_code}")


if __name__ == "__main__":
    main()