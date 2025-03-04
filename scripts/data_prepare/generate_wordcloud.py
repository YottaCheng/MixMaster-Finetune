"""
生成词云脚本 v1.2
（移除外部文件依赖，内置停用词库）
"""

import os
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import json
import re

# 配置常量
INPUT_PATH = "/Volumes/Study/prj/data/raw/training_raw_data.docx"
WORDCLOUD_OUTPUT_PATH = "/Volumes/Study/prj/data/processed/wordcloud.png"
JSON_OUTPUT_PATH = "/Volumes/Study/prj/data/processed/top_20_words.json"

def load_text_from_docx(file_path):
    """从 .docx 文件中提取所有段落文本"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    try:
        doc = Document(file_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        raise RuntimeError(f"文档读取失败：{str(e)}")

def is_valid_word(word):
    """智能词汇验证（增强过滤逻辑）"""
    # 基础过滤条件
    if len(word) < 2 or not re.search(r'[\u4e00-\u9fa5]', word):
        return False
    
    # 无意义词汇黑名单（已内置）
    meaningless_words = {
        "一点", "些", "有些", "某个", "某些", "那种", 
        "这种", "这个", "那个", "哪些", "什么", "怎么"
    }
    
    # 模式匹配过滤
    quantifier_patterns = [
        r".*点$", r".*些$", r"某.*", r".*种$", r".*么$"
    ]
    
    return (
        word not in meaningless_words and
        not any(re.match(p, word) for p in quantifier_patterns)
    )

def preprocess_text(text):
    """预处理文本（内置停用词库）"""
    # 内置专业停用词库
    stopwords = {
        "的", "是", "在", "和", "有", "与", "了", "这", "我们", "可以",
        "要", "对", "就", "也", "都", "而", "及", "或", "但", "更", "其",
        "他", "它", "他们", "它们", "这个", "那个", "这些", "那些", "一种"
    }
    
    # 精确分词+过滤
    words = [
        word for word in jieba.lcut(text)
        if word not in stopwords and is_valid_word(word)
    ]
    
    return words

def generate_wordcloud(word_freq, output_path):
    """生成词云（优化显示参数）"""
    wc = WordCloud(
        font_path="/System/Library/Fonts/STHeiti Light.ttc",
        width=1200,
        height=600,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.85,
        collocations=False  # 禁用词组搭配
    )
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc.generate_from_frequencies(word_freq))
    plt.axis("off")
    plt.tight_layout()
    wc.to_file(output_path)

def save_top_words_to_json(word_freq, output_path, top_n=20):
    """保存高频词数据（带词频统计）"""
    top_words = {
        word: {
            "count": count,
            "percentage": f"{(count/sum(word_freq.values()))*100:.2f}%"
        } for word, count in word_freq.most_common(top_n)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_words, f, ensure_ascii=False, indent=2)

def main():
    try:
        # 加载并清洗文本
        raw_text = load_text_from_docx(INPUT_PATH)
        print(f"原始文本长度：{len(raw_text)}字符")
        
        # 预处理与统计
        words = preprocess_text(raw_text)
        print(f"有效词汇数量：{len(words)}")
        
        word_freq = Counter(words)
        print(f"不重复词数：{len(word_freq)}")
        
        # 生成可视化结果
        generate_wordcloud(word_freq, WORDCLOUD_OUTPUT_PATH)
        save_top_words_to_json(word_freq, JSON_OUTPUT_PATH)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()