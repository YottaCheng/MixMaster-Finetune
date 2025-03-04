import os
import joblib

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Please install deep-translator with: pip install deep-translator")
    import sys
    sys.exit(1)


class MixingLabelPredictor:
    def __init__(self, model_dir="../data/outputs"):
        # 加载训练好的模型和向量化器
        self.model = joblib.load(os.path.join(model_dir, "baseline_model.pkl"))
        self.vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        # 使用 deep-translator 的 GoogleTranslator，指定源语言为中文（简体）和目标语言为英文
        self.translator = GoogleTranslator(source='zh-CN', target='en')
        
        # 标签中英文映射（根据你的 label_mapping.csv 配置）
        self.label_mapping = {
            'High_Freq': ('高频提升', 'High Frequency Boost'),
            'Mid_Freq': ('中频调整', 'Mid Frequency Adjustment'),
            'Low_Freq': ('低频控制', 'Low Frequency Control'),
            'Compression': ('压缩处理', 'Compression'),
            'Reverb': ('混响效果', 'Reverb'),
            'Stereo_Width': ('声场宽度', 'Stereo Width'),
            'Volume': ('音量调节', 'Volume Control')
        }

    def translate_to_english(self, text):
        """将中文翻译为英文"""
        if self._contains_chinese(text):
            return self.translator.translate(text)
        return text

    def _contains_chinese(self, text):
        """检查是否包含中文字符"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def predict(self, input_text, lang="中文"):
        """执行预测，并返回一个包含三个值的元组：
           (中文标签, 英文标签, 标签代码)
        """
        # 1. 语言转换
        english_text = self.translate_to_english(input_text).lower()
        
        # 2. 向量化
        text_vector = self.vectorizer.transform([english_text])
        
        # 3. 预测
        predicted_label = self.model.predict(text_vector)[0]
        
        # 4. 获取中英文标签及代码
        predicted_label_zh = self.label_mapping[predicted_label][0]
        predicted_label_en = self.label_mapping[predicted_label][1]
        label_code = predicted_label
        
        # 返回中英文结果（调用者根据 lang 决定使用哪一个）
        return predicted_label_zh, predicted_label_en, label_code


def main():
    # 命令行交互测试
    predictor = MixingLabelPredictor()
    
    print("\n混音标签预测系统（输入 'exit' 退出）")
    while True:
        text = input("\n请输入混音需求描述（中英文均可）: ").strip()
        if text.lower() == 'exit':
            break
            
        # 使用元组解包获取返回结果
        predicted_label_zh, predicted_label_en, label_code = predictor.predict(text, lang="中文")
        print("\n=== 预测结果 ===")
        print(f"原始输入: {text}")
        print(f"中文标签: {predicted_label_zh}")
        print(f"英文标签: {predicted_label_en}")
        print(f"标签代码: {label_code}")


if __name__ == "__main__":
    main()