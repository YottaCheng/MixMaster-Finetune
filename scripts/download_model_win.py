from huggingface_hub import snapshot_download
import os

# 指定模型名称
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 指定本地存储路径
local_model_dir = "D:/kings/MixMaster-Finetune/models"
os.makedirs(local_model_dir, exist_ok=True)

# 下载完整模型文件
print("正在下载模型...")
downloaded_path = snapshot_download(repo_id=model_name, cache_dir=local_model_dir)

print(f"模型已成功下载并保存到: {downloaded_path}")