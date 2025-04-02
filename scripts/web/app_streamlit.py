
import sys
import os
import dashscope
from dashscope import Generation
import streamlit as st
import time

lang = st.session_state.get("lang", "English") if "lang" in st.session_state else "English"
st.set_page_config(
    page_title="Mix Master",
    page_icon="🎚️",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# ---------- 添加当前目录到导入路径 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 确保优先搜索当前目录

# ---------- 导入预测器 ----------
try:
    from predict import MixingLabelPredictor
    
    # 使用Streamlit缓存资源装饰器来加载模型
    @st.cache_resource
    def load_predictor(model_dir=r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v6"):
        print("加载模型中...这个过程只会执行一次")
        return MixingLabelPredictor(model_dir=model_dir)
    
    # 初始化预测器 - 现在模型只会加载一次
    try:
        # 使用默认路径
        predictor = load_predictor()
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        
        # 创建一个模拟预测器类
        class MockMixingLabelPredictor:
            def predict(self, text, lang):
                if lang == "中文":
                    return "高频提升", "声音空间感", "HF001"
                else:
                    return "High Frequency Enhancement", "Spatial Depth", "HF001"
        
        predictor = MockMixingLabelPredictor()
        st.warning("使用模拟预测器替代。可能是系统内存不足，请尝试增加虚拟内存或关闭其他应用程序。")
except ImportError as e:
    st.error(f"导入 MixingLabelPredictor 失败: {str(e)}")
    
    # 创建一个模拟预测器类
    class MockMixingLabelPredictor:
        def predict(self, text, lang):
            if lang == "中文":
                return "高频提升", "声音空间感", "HF001"
            else:
                return "High Frequency Enhancement", "Spatial Depth", "HF001"
    
    predictor = MockMixingLabelPredictor()
    st.warning("使用模拟预测器替代。请确保 predict.py 文件存在于同一目录中。")

# ---------- API配置 ----------
dashscope.api_key = "sk-3b986ed51abb4ed18aadde5d41e11397"

# ---------- 界面文本配置 ----------
UI_TEXTS = {
    "中文": {
        "title": "🎚️ Mix Master",
        "subtitle": "专业音频处理助手",
        "input_label": "请输入音频处理需求（中英文均可）",
        "output_label": "预测标签",
        "output_code": "标签代码",
        "advice_title": "混音建议",
        "edit_hint": "可直接编辑此文本",
        "copy_btn": "📋 复制到剪贴板",
        "copy_success": "✅ 已复制",
        "paste_placeholder": "[粘贴内容将显示在这里]",
        "paste_btn": "📝 手动粘贴",
        "clear_btn": "🧹 清空",
        "analyze_btn": "🚀 开始分析",
        "switch_lang": "🌐 switch to Eng",
        "examples_title": "💡 示例输入",
        "examples": [
            "人声高频需要更明亮",
            "增加鼓组的冲击力",
            "整体空间感不足"
        ],
        "error_msg": "⚠️ 预测失败：",
        "generating": "正在生成混音建议...",
        "api_error": "⚠️ 生成建议失败：",
        "paste_toolbox": "复制/粘贴工具箱",
        "toolbox_title": "工具箱",
        "footer": "© 2025 E.Stay Mix Master | 专业音频解决方案",
        "powered_by": "基于人工智能技术",
        "input_section": "输入需求",
        "output_section": "分析结果",
        "advice_section": "混音建议",
        "examples_section": "示例输入", 
        "advice_placeholder": "点击'开始分析'生成混音建议..."
    },
    "English": {
        "title": "🎚️ Mix Master",
        "subtitle": "Professional Audio Processing Assistant",
        "input_label": "Enter audio processing request (Chinese/English)",
        "output_label": "Predicted Label",
        "output_code": "Label Code",
        "advice_title": "Mixing Advice",
        "edit_hint": "You can edit this text directly",
        "copy_btn": "📋 Copy to Clipboard",
        "copy_success": "✅ Copied",
        "paste_placeholder": "[Pasted content will appear here]",
        "paste_btn": "📝 Manual Paste", 
        "clear_btn": "🧹 Clear",
        "analyze_btn": "🚀 Analyze",
        "switch_lang": "🌐 Switch to Chinese",
        "examples_title": "💡 Example Inputs",
        "examples": [
            "Vocals need more brightness",
            "Increase drum punchiness",
            "Lack of overall spatial depth"
        ],
        "error_msg": "⚠️ Prediction failed: ",
        "generating": "Generating mixing advice...",
        "api_error": "⚠️ Failed to generate advice: ",
        "paste_toolbox": "Copy/Paste Toolbox",
        "toolbox_title": "Toolbox",
        "footer": "© 2025 E.Stay Mix Master | Professional Audio Solution",
        "powered_by": "Powered by AI Technology",
        "input_section": "Input Request",
        "output_section": "Analysis Results",
        "advice_section": "Mixing Advice",
        "examples_section": "Example Inputs",
        "advice_placeholder": "Click 'Analyze' to generate mixing advice..."
    }
}

# ---------- 设置页面配置 (MUST BE THE FIRST STREAMLIT COMMAND) ----------


# 立即调用设置页面配置的函数


# ---------- 核心逻辑 ----------
def get_mixing_advice(user_input, label, lang="中文"):
    """调用DashScope API生成混音建议"""
    try:
        # 构建提示词
        if lang == "中文":
            prompt = f"""
我是一位专业的音频混音工程师。我想要实现以下效果："{user_input}"
根据分析，这属于"{label}"类型的处理需求。

请给出简短的混音建议，介绍如何调整音频参数实现这个效果。
不要提及具体的插件名称，也不要给出具体的数值参数，只需提供调整方向和技术思路。
建议应简洁明了，不超过3-4句话。用中文回答。
"""
        else:
            prompt = f"""
I am a professional audio mixing engineer. I want to achieve the following effect: "{user_input}"
Based on analysis, this belongs to the "{label}" type of processing requirement.

Please provide brief mixing advice on how to adjust audio parameters to achieve this effect.
Do not mention specific plugin names, and do not provide specific numerical parameters, just provide adjustment directions and technical approach.
The advice should be concise, no more than 3-4 sentences. Answer in English.
"""

        # 显示进度信息
        progress_placeholder = st.empty()
        progress_placeholder.markdown(f"**{UI_TEXTS[lang]['generating']}**")
        
        try:
            # 调用DashScope API - 尝试流式响应
            response = Generation.call(
                model="qwq-plus-2025-03-05",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                incremental_output=True,
                result_format="message",
                temperature=0.2,
                top_p=0.7,
                max_tokens=150
            )
            
            # 解析流式响应
            full_response = []
            progress_text = ""
            
            for chunk in response:
                if chunk.status_code == 200:
                    if hasattr(chunk.output, 'choices') and len(chunk.output.choices) > 0:
                        message = chunk.output.choices[0].get('message')
                        if message and 'content' in message:
                            content = message['content']
                            full_response.append(content)
                            progress_text += content
                            progress_placeholder.markdown(f"**{UI_TEXTS[lang]['generating']}**\n\n{progress_text}")
                else:
                    progress_placeholder.empty()
                    return f"{UI_TEXTS[lang]['api_error']} 错误码 {chunk.status_code}"
            
            progress_placeholder.empty()
            return ''.join(full_response).strip()
            
        except Exception as stream_error:
            # 如果流式响应失败，回退到非流式响应
            try:
                # 非流式响应
                response = Generation.call(
                    model="qwq-plus-2025-03-05",
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    result_format="message",
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=150
                )
                
                progress_placeholder.empty()
                
                # 解析非流式响应
                if response.status_code == 200:
                    if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                        message = response.output.choices[0].get('message')
                        if message and 'content' in message:
                            return message['content'].strip()
                
                return f"{UI_TEXTS[lang]['api_error']} 错误码 {response.status_code}"
            
            except Exception as e:
                progress_placeholder.empty()
                return f"{UI_TEXTS[lang]['api_error']} {str(e)}"
            
    except Exception as e:
        return f"{UI_TEXTS[lang]['api_error']}{str(e)}"

def predict_wrapper(text, lang):
    """预测函数包装器"""
    try:
        # 获取预测结果，始终同时获取中英文结果
        zh_main, zh_secondary, code = predictor.predict(text, "中文")
        en_main, en_secondary, _ = predictor.predict(text, "English")
        
        # 获取混音建议
        # 使用当前语言对应的标签获取建议
        advice = get_mixing_advice(
            text, 
            zh_main if lang == "中文" else en_main, 
            lang
        )
        
        # 根据语言选择展示方式
        if lang == "中文":
            full_label = zh_main
            if zh_secondary:
                full_label += "，" + zh_secondary
            return full_label, code, advice
        else:
            full_label = en_main
            if en_secondary:
                full_label += ", " + en_secondary
            return full_label, code, advice
    except Exception as e:
        error_msg = f"{UI_TEXTS[lang]['error_msg']}{str(e)}"
        return error_msg, "ERROR", ""
        
# ---------- Streamlit UI 定义 ----------



# 设置自定义CSS样式
def set_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* 主要样式变量 */
    :root {
        --primary-color: #7c3aed;
        --primary-light: #a78bfa;
        --primary-dark: #6d28d9;
        --primary-gradient: linear-gradient(135deg, #7c3aed, #5b21b6);
        
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --card-secondary-bg: #f1f5f9;
        
        --text-color: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        
        --border-color: #e2e8f0;
        --radius-md: 10px;
    }

    /* 主样式 */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 标题样式 */
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 26px;
        margin-bottom: 0;
        color: var(--text-color);
        position: relative;
        display: inline-block;
    }
    
    .app-title::after {
        content: "";
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 40px;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: var(--text-secondary);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* 区块标题样式 */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 17px;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 20px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 4px solid var(--primary-color);
        line-height: 1.2;
    }
    
    /* 按钮样式 */
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: var(--radius-md);
        transition: all 0.2s ease;
    }
    
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--primary-gradient);
        border: none;
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 示例按钮样式 */
    .example-button {
        background-color: var(--card-secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 8px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 14px;
        display: inline-block;
    }
    
    .example-button:hover {
        background-color: white;
        border-color: var(--primary-color);
        transform: translateY(-1px);
    }
    
    /* 卡片和容器样式 */
    .output-container {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 15px;
        margin: 10px 0;
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
    }
    
    /* 文本输入框样式 */
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        font-family: 'Inter', sans-serif;
    }
    
    /* 工具箱样式 */
    .toolbox {
        margin-top: 30px;
    }
    
    /* 提示文本样式 */
    .stCaption {
        color: var(--text-muted);
        font-size: 13px;
        font-style: italic;
        text-align: right;
    }
    
    /* 修改滑动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--card-bg);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    
    /* 针对移动端的响应式调整 */
    @media (max-width: 768px) {
        .app-title {
            font-size: 22px;
        }
        
        .app-subtitle {
            font-size: 14px;
        }
        
        .section-title {
            font-size: 16px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 创建标题和副标题
def render_header():
    lang = st.session_state.get("lang", "中文")
    st.markdown(f'<h1 class="app-title">{UI_TEXTS[lang]["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{UI_TEXTS[lang]["subtitle"]}</p>', unsafe_allow_html=True)

# 创建示例部分
def render_examples():
    lang = st.session_state.get("lang", "中文")
    
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["examples_section"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p>{UI_TEXTS[lang]["examples_title"]}</p>', unsafe_allow_html=True)
    
    # 使用列布局创建示例按钮
    cols = st.columns(3)
    for i, example in enumerate(UI_TEXTS[lang]["examples"]):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.user_input = example
                st.session_state.run_analysis = True
                st.rerun()

# 创建输入部分
def render_input_section():
    lang = st.session_state.get("lang", "中文")
    
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["input_section"]}</div>', unsafe_allow_html=True)
    
    # 设置初始值
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # 创建输入框
    user_input = st.text_area(
        UI_TEXTS[lang]["input_label"],
        value=st.session_state.user_input,
        height=100,
        placeholder=UI_TEXTS[lang]["examples"][0],
        key="user_input_widget"
    )
    
    # 同步输入框值到session_state
    st.session_state.user_input = user_input
    
    # 功能按钮行
    cols = st.columns([1, 1, 1])
    
    with cols[0]:
        analyze_clicked = st.button(
            UI_TEXTS[lang]["analyze_btn"],
            type="primary",
            use_container_width=True,
            key="analyze_button"
        )
    
    with cols[1]:
        clear_clicked = st.button(
            UI_TEXTS[lang]["clear_btn"],
            type="secondary",
            use_container_width=True,
            key="clear_button"
        )
    
    with cols[2]:
        lang_clicked = st.button(
            UI_TEXTS[lang]["switch_lang"],
            type="secondary",
            use_container_width=True,
            key="lang_button"
        )
    
    # 处理清空按钮事件
    if clear_clicked:
        st.session_state.user_input = ""
        st.session_state.prediction_label = ""
        st.session_state.prediction_code = ""
        st.session_state.advice = ""
        st.rerun()
    
    # 处理语言切换事件
    if lang_clicked:
        st.session_state.lang = "English" if lang == "中文" else "中文"
        st.rerun()
    
    return user_input, analyze_clicked

# 创建输出部分
def render_output_section():
    lang = st.session_state.get("lang", "中文")
    
    # 获取预测结果
    prediction_label = st.session_state.get("prediction_label", "")
    prediction_code = st.session_state.get("prediction_code", "")
    advice = st.session_state.get("advice", "")
    
    # 只有当有预测结果时才显示输出区域
    if prediction_label or prediction_code or advice:
        st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["output_section"]}</div>', unsafe_allow_html=True)
        
        # 预测标签和代码
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(UI_TEXTS[lang]["output_label"], value=prediction_label, disabled=True)
        with col2:
            st.text_input(UI_TEXTS[lang]["output_code"], value=prediction_code, disabled=True)
        
        # 混音建议
        st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["advice_section"]}</div>', unsafe_allow_html=True)
        
        # 可编辑建议
        edited_advice = st.text_area(
            label="",
            value=advice if advice else UI_TEXTS[lang]["advice_placeholder"],
            height=150,
            key="advice_widget"
        )
        
        # 同步编辑后的建议
        if edited_advice != advice and advice:
            st.session_state.advice = edited_advice
        
        st.caption(UI_TEXTS[lang]["edit_hint"])
        
        # 添加复制按钮
        if advice:
            copy_clicked = st.button(
                UI_TEXTS[lang]["copy_btn"],
                key="copy_button"
            )
            
            # 处理复制事件
            if copy_clicked:
                # 将建议添加到剪贴板区域
                st.session_state.clipboard_content = edited_advice
                
                # 显示复制成功消息
                st.success(UI_TEXTS[lang]["copy_success"])

# 创建工具箱部分
def render_toolbox():
    lang = st.session_state.get("lang", "中文")
    
    with st.expander(UI_TEXTS[lang]["toolbox_title"]):
        # 初始化剪贴板内容
        if "clipboard_content" not in st.session_state:
            st.session_state.clipboard_content = ""
        
        # 剪贴板区域
        clipboard_content = st.text_area(
            UI_TEXTS[lang]["paste_btn"],
            value=st.session_state.clipboard_content,
            placeholder=UI_TEXTS[lang]["paste_placeholder"],
            height=100,
            key="clipboard_widget"
        )
        
        # 同步剪贴板内容并自动应用到输入框（Gradio风格）
        if st.session_state.get("clipboard_content") != clipboard_content and clipboard_content:
            st.session_state.clipboard_content = clipboard_content
            st.session_state.user_input = clipboard_content
            st.rerun()
        
        st.caption(UI_TEXTS[lang]["edit_hint"])
        
        # 按钮完全移除

# 创建页脚
def render_footer():
    lang = st.session_state.get("lang", "中文")
    
    st.markdown(f"""
    <div class="footer">
        {UI_TEXTS[lang]['footer']}
        <div style="margin-top: 5px; font-size: 12px;">
            {UI_TEXTS[lang]['powered_by']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# 主应用函数
def main():
    # 初始化 session_state
    if "lang" not in st.session_state:
        st.session_state.lang = "English"  # 默认语言改为英文
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
    
    set_custom_css()
    
    # 渲染页面组件
    render_header()
    render_examples()
    
    # 输入区域
    user_input, analyze_clicked = render_input_section()
    
    # 处理分析按钮事件
    if (analyze_clicked and user_input) or (st.session_state.run_analysis and st.session_state.user_input):
        # 重置运行标志
        st.session_state.run_analysis = False
        
        with st.spinner(UI_TEXTS[st.session_state.lang]["generating"]):
            # 执行预测
            label, code, advice = predict_wrapper(st.session_state.user_input, st.session_state.lang)
            
            # 保存结果到session_state
            st.session_state.prediction_label = label
            st.session_state.prediction_code = code
            st.session_state.advice = advice
    
    # 输出区域
    render_output_section()
    
    # 工具箱
    render_toolbox()
    
    # 页脚
    render_footer()

# 替代预测器类
class MockMixingLabelPredictor:
    def predict(self, text, lang):
        if lang == "中文":
            return "高频提升", "声音空间感", "HF001"
        else:
            return "High Frequency Enhancement", "Spatial Depth", "HF001"

# 应用程序入口
if __name__ == "__main__":
    main()