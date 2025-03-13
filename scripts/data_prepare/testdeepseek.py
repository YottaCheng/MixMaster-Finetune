import dashscope
from dashscope import Generation

def call_qwq_model():
    """符合QwQ模型特性的流式调用方法"""
    dashscope.api_key = "sk-3b986ed51abb4ed18aadde5d41e11397"  # 替换有效API Key
    
    try:
        # 必须配置的参数组合
        response = Generation.call(
            model="qwq-plus-2025-03-05",  # 确认模型名称准确
            messages=[{"role": "user", "content": "你是qwq-plus还是Qwen-Max"}],
            stream=True,                # 强制启用流式
            incremental_output=True,    # 必须保持默认True（不可修改）
            result_format="message",    # 必须保持默认message格式
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024
        )

        print("🔄 开始接收流式响应...")
        full_response = ""
        for chunk in response:
            if chunk.status_code == 200:
                content = chunk.output.choices[0]['message']['content']
                full_response += content
                print(content, end="", flush=True)  # 实时显示增量内容
            else:
                print(f"\n⚠️ 异常数据块 | Code: {chunk.code} | Msg: {chunk.message}")

        print("\n\n=== 完整响应 ===")
        print(full_response)

    except Exception as e:
        print(f"全局异常: {str(e)}")

if __name__ == "__main__":
    call_qwq_model()