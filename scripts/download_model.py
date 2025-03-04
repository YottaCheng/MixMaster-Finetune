from transformers import AutoModel, AutoTokenizer

# 下载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
cache_dir = "/Volumes/Study/prj/huggingface_cache"

model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    force_download=True,  # 强制重新下载
    resume_download=False
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    force_download=True,
    resume_download=False
)

print(f"✅ 模型已下载到: {cache_dir}")