import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# 配置路径
MODEL_PATH = "/Volumes/Study/prj/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"

def test_model_loading():
    print("开始测试本地模型加载...")
    
    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型路径不存在 - {MODEL_PATH}")
        return
    
    print(f"模型路径检查通过: {MODEL_PATH}")
    
    try:
        # 加载配置
        print("正在加载模型配置...")
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("配置加载成功:", config)
        
        # 处理 torch_dtype 类型
        if isinstance(config.torch_dtype, str):
            if hasattr(torch, config.torch_dtype):
                torch_dtype = getattr(torch, config.torch_dtype)
            else:
                torch_dtype = torch.bfloat16  # 默认类型
        else:
            torch_dtype = config.torch_dtype
        
        # 检查是否使用 MPS 并回退到 float32
        if torch.backends.mps.is_available() and torch_dtype == torch.bfloat16:
            print("警告：MPS 不支持 bfloat16，回退到 float32")
            torch_dtype = torch.float32

        print(f"torch_dtype 设置为: {torch_dtype}")
        
        # 加载分词器
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # 手动设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"已将 pad_token 设置为: {tokenizer.pad_token}")
        else:
            print(f"pad_token 已存在: {tokenizer.pad_token}")
        
        print("分词器加载成功")
        
        # 加载模型
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            config=config,
            device_map="cpu",  # 强制使用 CPU
            torch_dtype=torch_dtype,  # 动态设置数据类型
            trust_remote_code=True
        )
        model.eval()
        print("模型加载成功")
        
        # 测试生成
        print("正在测试模型生成...")
        prompts = [
            "将以下内容转换为中文：Hello, this is a test input.",
            "计算 123 加 456 的结果。",
            "解释什么是人工智能。",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n测试 {i}: 提示词: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,  # 显式传递 attention_mask
                    max_new_tokens=100,  # 严格限制生成长度
                    temperature=0.4,   # 降低随机性
                    top_p=0.7,         # 控制多样性
                    repetition_penalty=1.5,  # 更强避免重复内容
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,  # 确保生成结束
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 后处理：去除提示词、重复内容和冗余信息
            generated_text = re.sub(r"(?i)" + re.escape(prompt), "", generated_text)  # 移除提示词
            generated_text = re.sub(r"\s+", " ", generated_text).strip()  # 去除多余空格
            generated_text = re.sub(r"[^\w\s，。？！]", "", generated_text)  # 移除特殊字符
            generated_text = re.sub(r"(think|逐步思考过程|解答).*", "", generated_text, flags=re.IGNORECASE)  # 移除无关内容

            print("生成结果:")
            print(generated_text)
        
    except Exception as e:
        import traceback
        print("模型加载或生成失败:")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()