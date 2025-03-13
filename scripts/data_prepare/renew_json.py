"""
论文参考:
1. 《A Survey of Data Augmentation Approaches for NLP》
2. 《AEDA: An Easier Data Augmentation Technique for Text Classification》
"""
import os
import json
from pathlib import Path
from dashscope import Generation

# 配置相对路径
BASE_DIR = Path(__file__).parent.parent.parent  # 假设脚本在 scripts/ 目录
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

top_words_file = DATA_DIR / "processed" / "top_20_words.json"
synonyms_file = CONFIG_DIR / "music_synonyms.json"

def load_json(file_path):
    """加载 JSON 文件（增加路径校验）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"关键文件缺失: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"配置文件损坏: {file_path}")

def save_json(data, file_path):
    """保存 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_api_key():
    """从环境变量加载 API Key"""
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("未找到 API Key，请设置环境变量 QWEN_API_KEY")
    return api_key

def call_qwen_api(prompt: str, api_key: str, model: str = "qwen-plus"):
    """调用 Qwen API 并返回结果"""
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            api_key=api_key
        )
        return response
    except Exception as e:
        print(f"API 调用失败：{str(e)}")
        return None

def generate_synonyms(word, api_key):
    """生成缺失词的同义词或替代表达"""
    # 优化后的提示词：明确要求生成简短的JSON列表，最多4个词
    prompt = (
        f"假设你是一个音乐小白，你要向混音师提需求，其中有字眼： '{word}' 提供3个以内（含3个）的偏口语化的中文同义词或替代表达，"
        f"以严格JSON数组格式返回，不添加任何解释文字。例如：['同义词1', '同义词2']"
    )
    response = call_qwen_api(prompt, api_key)
    
    if response and "output" in response:
        try:
            # 直接解析为列表
            synonyms = json.loads(response["output"]["text"])
            # 确保返回的是列表且元素不超过4个
            if isinstance(synonyms, list) and len(synonyms) <= 4:
                return synonyms
            else:
                print(f"响应格式错误：期望列表且长度<=4，实际长度{len(synonyms)}")
                return []
        except json.JSONDecodeError:
            print(f"无法解析 API 响应为 JSON: {response['output']['text']}")
            return []
    return []

def main():
    # 加载 API Key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(e)
        return

    # 加载高频词和替换词词典
    top_words = load_json(top_words_file)
    synonyms_dict = load_json(synonyms_file)

    # 检查哪些高频词没有对应的替换词
    missing_words = []
    for word in top_words.keys():
        found = False
        for category, replacements in synonyms_dict.items():
            if word in replacements:
                found = True
                break
        if not found:
            missing_words.append(word)

    # 如果有缺失的词，创建一个新的类别 "新增词汇"
    if missing_words:
        if "新增词汇" not in synonyms_dict:
            synonyms_dict["新增词汇"] = {}

        # 为缺失的词生成替换词
        for word in missing_words:
            print(f"正在为 '{word}' 生成替换词...")
            new_synonyms = generate_synonyms(word, api_key)
            if new_synonyms:
                # 将新生成的替换词添加到 "新增词汇" 类别中
                synonyms_dict["新增词汇"][word] = new_synonyms
                print(f"已为 '{word}' 添加替换词: {new_synonyms}")

    # 保存更新后的替换词词典
    save_json(synonyms_dict, synonyms_file)
    print("替换词词典已更新并保存。")

if __name__ == "__main__":
    main()