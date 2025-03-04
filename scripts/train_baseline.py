import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

# 路径配置
MODEL_PATH = "/Volumes/Study/prj/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"
DATA_PATH = "/Volumes/Study/prj/data/processed/train_data_en.csv"
MODEL_SAVE_DIR = "/Volumes/Study/prj/data/outputs"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, output_dict=True)
    return {
        "accuracy": acc,
        "f1": f1,
        "classification_report": report
    }

def main():
    # 1. 加载数据
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 数据加载成功，共 {len(df)} 条样本")
    print("标签分布:\n", df["label_en"].value_counts())

    # 2. 数据预处理
    X = df["text_en"].fillna("").str.lower()  # 转小写统一处理
    y = df["label_en"]

    # 将标签转换为数值
    label_map = {label: idx for idx, label in enumerate(y.unique())}
    y = y.map(label_map)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n数据集划分：训练集 {len(X_train)} 条 | 测试集 {len(X_test)} 条")

    # 转换为 Hugging Face Dataset 格式
    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))

    # 3. 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 确保分词器有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=len(label_map)
    )

    # 确保模型的嵌入层包含 pad_token 对应的嵌入向量
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))  # 调整嵌入层大小

    # 4. 数据预处理：Tokenization
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,  # 减少最大长度以节省内存
            return_tensors="pt"
        )

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置格式为 PyTorch 张量
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 5. 微调模型
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        eval_strategy="epoch",  # 使用 eval_strategy 替代 evaluation_strategy
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # 进一步减少批量大小
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 使用梯度累积模拟更大批量
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # fp16=True,  # 禁用 FP16，因为 Mac 不支持
        dataloader_num_workers=0,  # 使用单线程数据加载器
        remove_unused_columns=False,  # 保留所有列
        report_to="none",  # 禁用日志报告
        device="cpu",  # 强制使用 CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 6. 评估模型
    eval_results = trainer.evaluate()
    print("\n=== 测试集性能 ===")
    print("准确率：", eval_results["eval_accuracy"])
    print("宏平均F1：", eval_results["eval_f1"])
    print("\n分类报告：\n", eval_results["eval_classification_report"])

    # 7. 保存模型
    trainer.save_model(os.path.join(MODEL_SAVE_DIR, "fine_tuned_model"))
    tokenizer.save_pretrained(os.path.join(MODEL_SAVE_DIR, "fine_tuned_tokenizer"))
    print(f"\n💾 模型已保存至 {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()