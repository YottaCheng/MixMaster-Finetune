import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# 配置路径
MODEL_PATH = "/Volumes/Study/prj/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"
DATA_PATH = "/Volumes/Study/prj/config/fine_tune_data.json"  # 微调数据文件路径

def fine_tune_model():
    print("开始微调模型...")
    
    # 强制使用 CPU 并释放 GPU 缓存
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device_map = "cpu"
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型并启用梯度检查点
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.config.use_cache = False  # 显式禁用缓存
    model.gradient_checkpointing_enable()  # 启用梯度检查点
    
    # 加载微调数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 格式化数据
    formatted_data = []
    for item in data:
        prompt = item["prompt"]
        response = item["response"]
        full_text = f"{prompt} {response}{tokenizer.eos_token}"
        formatted_data.append({"text": full_text})
    
    # 只使用部分数据进行测试
    dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data[:1]]})
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)  # 缩短最大长度
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 减少批量大小
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        save_steps=100,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=False,  # 不启用混合精度
        push_to_hub=False
    )
    
    # 使用 Trainer 进行微调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # 开始微调
    trainer.train()
    
    # 保存微调后的模型
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("微调完成，模型已保存到 ./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()