import json
import os
import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer



MODEL_DIR = "/Volumes/Study/prj/huggingface_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True,  # 使用快速分词器
    add_special_tokens=False  # 避免动态添加特殊标记
)
def validate_data(log_path, json_path, label_mapping_path):
    """主验证流程"""
    create_backup(json_path)
    if not validate_files(log_path, json_path, label_mapping_path):
        return False
    log_data, cleaned_data, valid_labels = load_data(log_path, json_path, label_mapping_path)
    if not all([log_data, cleaned_data, valid_labels]):
        return False
    model_valid = validate_model(MODEL_DIR)
    print(f"\n🔧 模型验证结果: {'✅ 通过' if model_valid else '⚠️ 警告（可尝试继续）'}")
    check_result = intelligent_sampling_check(cleaned_data, json_path)
    generate_final_report(log_data, cleaned_data, model_valid, check_result)
    return check_result and model_valid

def validate_files(*paths):
    """验证文件存在性"""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print(f"❌ 缺失文件: {', '.join(missing)}")
        return False
    return True

def load_data(log_path, json_path, label_mapping_path):
    """加载所有数据"""
    try:
        with open(log_path) as f:
            log_data = json.load(f)
        with open(json_path) as f:
            cleaned_data = json.load(f)
        with open(label_mapping_path, encoding='utf-8') as f:
            valid_labels = {row['中文标签'].strip() for row in csv.DictReader(f)}
        return log_data, cleaned_data, valid_labels
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return None, None, None

def validate_model(model_dir):
    """增强型模型验证（包含功能性测试）"""
    essential_files = {'config.json', 'model.safetensors'}  # 替换 pytorch_model.bin 为 model.safetensors
    try:
        model_path = Path(model_dir)
        existing_files = {f.name for f in model_path.glob('*')}
        missing = essential_files - existing_files
        
        if missing:
            print(f"❌ 缺失关键文件: {', '.join(missing)}")
            return False
        
        # 功能性验证
        print("⏳ 正在进行模型加载测试...", end="")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                use_fast=True,  # 使用快速分词器
                add_special_tokens=False  # 避免动态添加特殊标记
            )
            model = AutoModelForCausalLM.from_pretrained(model_dir)  # 加载支持生成任务的模型
        except Exception as e:
            print(f"\n❌ 模型加载失败: {str(e)}")
            return False
        
        # 文本生成测试
        print("\n⏳ 正在进行文本生成测试...", end="")
        test_input = "你好，我是Qwen"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        try:
            outputs = model.generate(**inputs, max_new_tokens=10)  # 调用 .generate()
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n✅ 生成测试结果: {generated_text}")
            return True
        except Exception as e:
            print(f"\n❌ 文本生成失败: {str(e)}")
            return False

    except Exception as e:
        print(f"❌ 模型验证异常: {str(e)}")
        return False
    
def intelligent_sampling_check(data, json_path, sample_size=10):
    """智能抽样检查与自动清理"""
    print(f"\n🔍 开始智能抽样检查（共{sample_size}条）:")
    
    # 直接随机抽样（注释可疑样本检测）
    check_samples = random.sample(data, sample_size)
    
    # 执行检查
    invalid_ids = process_samples(check_samples, data)
    
    # 自动清理数据
    cleanup_data(json_path, data, invalid_ids)
        
    return len(invalid_ids) == 0

def detect_suspicious_samples(data):
    """检测可疑样本"""
    suspicious = []
    seen_texts = set()
    
    for item in data:
        # 空文本检测
        if len(item['text'].strip()) < 3:
            suspicious.append(item)
            continue
            
        # 重复内容检测
        text = item['text'].strip().lower()
        if text in seen_texts:
            suspicious.append(item)
        else:
            seen_texts.add(text)
    
    return suspicious

def select_samples(data, suspicious, sample_size):
    """选择检查样本"""
    # 优先选择可疑样本
    check_samples = suspicious[:sample_size//2] if suspicious else []
    
    # 补充随机样本
    remaining = sample_size - len(check_samples)
    if remaining > 0:
        check_samples += random.sample(data, remaining)
    
    # 去重处理
    return list({item['id']: item for item in check_samples}.values())[:sample_size]

def process_samples(samples, full_data):
    """处理样本检查"""
    invalid_ids = set()
    
    for idx, item in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] ID:{item['id']}")
        print(f"标签: {item['label']}")
        print(f"内容: {item['text'][:150]}...")
        
        # 移除自动重复检测（已注释）
        # if is_duplicate(item, full_data):
        #     print("⚠️ 检测到重复内容！")
        
        # 简化交互逻辑
        choice = input("是否保留该样本？(y-保留/n-删除): ").lower()
        if choice == 'n':
            invalid_ids.add(item['id'])
        elif choice != 'y':
            print("输入错误，默认保留")
    
    return invalid_ids

def is_duplicate(item, data):
    """检查重复内容"""
    return sum(1 for x in data if x['text'].strip().lower() == item['text'].strip().lower()) > 1

def cleanup_data(json_path, data, invalid_ids):
    """清理无效数据"""
    valid_data = [item for item in data if item['id'] not in invalid_ids]
    
    try:
        with open(json_path, 'w') as f:
            json.dump(valid_data, f, indent=2)
        print(f"\n♻️ 已删除 {len(invalid_ids)} 条无效数据")
    except Exception as e:
        print(f"❌ 数据清理失败: {str(e)}")

def create_backup(file_path):
    """创建数据备份"""
    backup_path = f"{file_path}.bak"
    if not Path(backup_path).exists():
        try:
            shutil.copy(file_path, backup_path)
        except Exception as e:
            print(f"⚠️ 备份失败: {str(e)}")

def generate_final_report(log_data, data, model_valid, check_result):
    """改进后的最终报告"""
    print("\n📝 最终验证报告:")
    print(f"- 数据总量: {len(data)} 条 (日志记录: {log_data['summary']['total_samples']})")
    print(f"- 模型验证: {'✅ 正常' if model_valid else '⚠️ 缺失关键文件'}")
    print(f"- 样本检查: {'✅ 通过' if check_result else '⚠️ 需要进一步检查'}")
    print(f"- 推荐操作: {'' if check_result and model_valid else '请重新运行验证'}")

if __name__ == "__main__":
    log_path = "/Volumes/Study/prj/data/processed/data_quality_report.json"
    json_path = "/Volumes/Study/prj/data/processed/cleaned_data.json"
    label_mapping = "/Volumes/Study/prj/data/raw/label_mapping.csv"
    
    validate_data(log_path, json_path, label_mapping)