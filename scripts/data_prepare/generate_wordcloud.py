"""
Generate Word Cloud Script v1.4
(Fix code block markup errors + full path validation)
References:
1. 《A Survey of Data Augmentation Approaches for NLP》
2. 《AEDA: An Easier Data Augmentation Technique for Text Classification》
"""
from pathlib import Path
import os
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import json
import re

# ==================== Configuration of relative paths ====================
BASE_DIR = Path(__file__).parent.parent.parent  # Assuming script is in scripts/data_prepare/ directory
CONFIG_PATH = BASE_DIR / "config" / "music_synonyms.json"
INPUT_PATH = BASE_DIR / "data" / "raw" / "training_raw_data.docx"
WORDCLOUD_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "wordcloud.png"
JSON_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "top_20_words.json"

# ==================== Function definitions ====================
def load_text_from_docx(file_path):
    """Load text content from .docx file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    try:
        doc = Document(file_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        raise RuntimeError(f"Document loading failed: {str(e)}")

def load_protected_terms(config_path):
    """Load protected domain terms from configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("_protected", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load protected terms config - {str(e)}")
        return []

def is_valid_word(word):
    """Smart word validation (enhanced filtering logic)"""
    # Basic filtering conditions
    if len(word) < 1 or not re.search(r'[a-zA-Z\u4e00-\u9fa5]', word):
        return False
    
    # Meaningless words blacklist (built-in)
    meaningless_words = {
        "一点", "些", "有些", "某个", "某些", "那种", 
        "这种", "这个", "那个", "哪些", "什么", "怎么",
        "one", "some", "this", "that", "those", "what", "how"
    }
    
    # Pattern matching filter
    quantifier_patterns = [
        r".*点$", r".*些$", r"某.*", r".*种$", r".*么$",
        r".*one$", r".*some$", r".*this$", r".*that$", r".*those$", r".*what$", r".*how$"
    ]
    
    return (
        word not in meaningless_words and
        not any(re.match(p, word) for p in quantifier_patterns)
    )

def preprocess_text(text):
    """Preprocess text (built-in stopword library + domain term protection)"""
    # Built-in professional stopword library
    stopwords = {
        "的", "是", "在", "和", "有", "与", "了", "这", "我们", "可以",
        "要", "对", "就", "也", "都", "而", "及", "或", "但", "更", "其",
        "他", "它", "他们", "它们", "这个", "那个", "这些", "那些", "一种",
        "a", "an", "the", "and", "or", "but", "for", "of", "to", 
        "in", "on", "at", "by", "with", "as", "from", "about", "into", 
        "through", "during", "before", "after", "above", "below", "up", 
        "down", "in", "out", "over", "under", "again", "further", "then", 
        "once", "here", "there", "when", "where", "why", "how", "all", "any", 
        "both", "each", "few", "more", "most", "other", "some", "such", "no", 
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
        "t", "can", "will", "just", "don", "should", "now"
    }
    
    # Load domain protected terms and add to stopwords
    protected_terms = load_protected_terms(CONFIG_PATH)
    stopwords.update(protected_terms)
    
    # Precise segmentation + filtering
    words = [
        word for word in jieba.lcut(text)
        if word not in stopwords and is_valid_word(word)
    ]
    
    return words

def generate_wordcloud(word_freq, output_path):
    """Generate word cloud (optimized display parameters)"""
    wc = WordCloud(
        font_path="/System/Library/Fonts/STHeiti Light.ttc",
        width=1200,
        height=600,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.85,
        collocations=False  # Disable bigram generation
    )
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc.generate_from_frequencies(word_freq))
    plt.axis("off")
    plt.tight_layout()
    wc.to_file(output_path)

def save_top_words_to_json(word_freq, output_path, top_n=20):
    """Save top words data (with frequency statistics)"""
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
        # Load and clean text
        raw_text = load_text_from_docx(INPUT_PATH)
        print(f"Raw text length: {len(raw_text)} characters")
        
        # Preprocess and statistics
        words = preprocess_text(raw_text)
        print(f"Valid words count: {len(words)}")
        
        word_freq = Counter(words)
        print(f"Unique words count: {len(word_freq)}")
        
        # Generate visualization results
        generate_wordcloud(word_freq, WORDCLOUD_OUTPUT_PATH)
        save_top_words_to_json(word_freq, JSON_OUTPUT_PATH)
        
        # Verify protected term isolation
        protected_terms = load_protected_terms(CONFIG_PATH)
        top_words = json.load(open(JSON_OUTPUT_PATH, "r", encoding="utf-8"))
        leakage = [term for term in protected_terms if term in top_words]
        if leakage:
            print(f"⚠️ Protected term leakage: {leakage}")
        else:
            print("✅ Domain term isolation successful")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()