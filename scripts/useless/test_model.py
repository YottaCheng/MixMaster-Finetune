from transformers import AutoModelForCausalLM, AutoTokenizer

# 替换为实际路径
model_dir = "/Volumes/Study/prj/huggingface_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"
# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="cpu"  # 强制使用 CPU
)

print("Model loaded successfully!")

# 测试生成标签
prompt = "根据以下描述生成一个音乐术语标签：整体人声可以稍微往后拖一点点"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("生成结果：", result.strip())