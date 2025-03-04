
# export QWEN_API_KEY="sk-3b986ed51abb4ed18aadde5d41e11397"
import os
from dashscope import Generation

# 配置 base_url
base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

def load_api_key():
    """从环境变量加载 API Key"""
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("未找到 API Key，请设置环境变量 QWEN_API_KEY")
    return api_key

def call_qwen_api(prompt: str, api_key: str, model: str = "qwen-max"):
    """调用 Qwen API 并返回结果"""
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url
        )
        return response
    except Exception as e:
        print(f"API 调用失败：{str(e)}")
        return None

def format_response(response):
    """格式化 API 返回结果"""
    if not response or "output" not in response:
        return "无效的响应"
    
    output = response["output"]
    usage = response.get("usage", {})
    
    formatted_output = (
        f"生成的文本：{output['text']}\n"
        f"Token 使用情况：\n"
        f"  输入 Tokens: {usage.get('input_tokens', '未知')}\n"
        f"  输出 Tokens: {usage.get('output_tokens', '未知')}\n"
        f"  总 Tokens: {usage.get('total_tokens', '未知')}\n"
        f"请求 ID：{response.get('request_id', '未知')}"
    )
    return formatted_output

def main():
    # 加载 API Key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(e)
        return
    
    # 第一次请求
    prompt1 = "你好，世界！"
    print("第一次请求：", prompt1)
    response1 = call_qwen_api(prompt1, api_key)
    if response1:
        print(format_response(response1))
    
    # 第二次请求（基于上下文或多轮对话）
    prompt2 = "你能帮我写一首诗吗？"
    print("\n第二次请求：", prompt2)
    response2 = call_qwen_api(prompt2, api_key)
    if response2:
        print(format_response(response2))

if __name__ == "__main__":
    main()