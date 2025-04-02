import sys
sys.path.append(r"D:\kings\prj\Finetune_local\LLaMA-Factory\src") 
from llamafactory.chat.chat_model import ChatModel
import argparse
from peft import PeftModel
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="D:/kings/prj/Finetune_local/Models/deepseek_R1_MixMaster/v6")
    parser.add_argument("--input_text", type=str, default="人声甜一点")
    args = parser.parse_args()

    # 参数配置（确保所有必要参数正确）
    args_dict = {
        "model_name_or_path": args.model_path,  # 基础模型路径
        #"adapter_name_or_path": "D:/kings/prj/Finetune_local/Models/deepseek_R1_MixMaster/v6",  # 关键参数
        "infer_backend": "huggingface",
        "template": "deepseek",
        "stage": "sft",
        "temperature": 0.3,
        "top_p": 0.7,
        "repetition_penalty": 1.1,
        "max_new_tokens": 20,
        "do_sample": True,
    }

    # 创建模型实例
    print("正在加载模型:", args.model_path)
    chat_model = ChatModel(args=args_dict)
    print("模型加载完成")

    # 优化后的提示词结构（关键修改）
    system_message = (
        "You are an audio processing expert. "
        "You must select 1-3 most relevant parameters from the given options. "
        "Only include parameters directly related to the input description."
    )

    user_prompt = f"""
    请根据以下音频处理需求，选择最相关的1-3个参数标签：
    
    输入文本：{args.input_text}
    
    可选参数：
    低频/中频/高频/reverb/效果器/声场/压缩/音量
    
    输出要求：
    1. 仅返回逗号分隔的标签（如：高频,压缩）
    2. 必须选择至少1个标签
    3. 优先选择最确定的标签
    4. 避免无关或模糊的选项
    """
    
    messages = [{"role": "user", "content": user_prompt.strip()}]

    # 生成回答
    print(f"正在处理输入: {args.input_text}")
    response = chat_model.chat(messages, system=system_message)
    
    # 处理输出（确保格式正确）
    output = response[0].response_text.strip()
    labels = [label.strip() for label in output.split(',') if label.strip()]

    # 打印结果
    print("="*50)
    print("输入文本:", args.input_text)
    print("原始响应:", output)
    print("处理后的标签:", ", ".join(labels))
    print("="*50)

if __name__ == "__main__":
    main()