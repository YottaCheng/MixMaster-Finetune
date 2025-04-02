"""
References:
1. 《A Survey of Data Augmentation Approaches for NLP》
2. 《AEDA: An Easier Data Augmentation Technique for Text Classification》
"""
import os
import json
from pathlib import Path
from dashscope import Generation

os.environ["QWEN_API_KEY"] = "sk-3b986ed51abb4ed18aadde5d41e11397"

# Configuration of relative paths
BASE_DIR = Path(__file__).parent.parent.parent  # Assuming script is in scripts/ directory
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

top_words_file = DATA_DIR / "processed" / "top_20_words.json"
synonyms_file = CONFIG_DIR / "music_synonyms.json"

def load_json(file_path):
    """Load JSON file (with path validation)"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file missing: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Corrupted configuration file: {file_path}")

def save_json(data, file_path):
    """Save JSON file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_api_key():
    """Load API Key from environment variables"""
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("API Key not found, please set environment variable QWEN_API_KEY")
    return api_key

def call_qwen_api(prompt: str, api_key: str, model: str = "qwen-plus"):
    """Call Qwen API and return result"""
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            api_key=api_key
        )
        return response
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None

def generate_synonyms(word, api_key):
    """Generate synonyms or alternative expressions for missing words"""
    # Optimized prompt: clearly request a short JSON list with up to 4 words
    prompt = (
        f"Assume you are a music novice and want to request something from a sound engineer. There is a term: '{word}'. "
        f"Provide 3 or fewer (inclusive) colloquial Chinese synonyms or alternative expressions, "
        f"in strict JSON array format without any explanatory text. For example: ['synonym1', 'synonym2']"
    )
    response = call_qwen_api(prompt, api_key)
    
    if response and "output" in response:
        try:
            # Directly parse as list
            synonyms = json.loads(response["output"]["text"])
            # Ensure it's a list and length is <= 4
            if isinstance(synonyms, list) and len(synonyms) <= 4:
                return synonyms
            else:
                print(f"Response format error: expected list with length <= 4, actual length {len(synonyms)}")
                return []
        except json.JSONDecodeError:
            print(f"Failed to parse API response as JSON: {response['output']['text']}")
            return []
    return []

def main():
    # Load API Key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(e)
        return

    # Load top words and synonyms dictionary
    top_words = load_json(top_words_file)
    synonyms_dict = load_json(synonyms_file)

    # Check which top words do not have corresponding synonyms
    missing_words = []
    for word in top_words.keys():
        found = False
        for category, replacements in synonyms_dict.items():
            if word in replacements:
                found = True
                break
        if not found:
            missing_words.append(word)

    # If there are missing words, create a new category "New Words"
    if missing_words:
        if "新增词汇" not in synonyms_dict:
            synonyms_dict["新增词汇"] = {}

        # Generate synonyms for missing words
        for word in missing_words:
            print(f"Generating synonyms for '{word}'...")
            new_synonyms = generate_synonyms(word, api_key)
            if new_synonyms:
                # Add newly generated synonyms to the "新增词汇" category
                synonyms_dict["新增词汇"][word] = new_synonyms
                print(f"Added synonyms for '{word}': {new_synonyms}")

    # Save the updated synonyms dictionary
    save_json(synonyms_dict, synonyms_file)
    print("Synonyms dictionary updated and saved.")

if __name__ == "__main__":
    main()