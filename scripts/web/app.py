import sys
import os
import dashscope
from dashscope import Generation
import streamlit as st

# ---------- 关键修复：直接导入本地模块 ----------
# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 直接导入同级目录中的 predict.py
# 由于文件就在同一个目录，使用相对导入
sys.path.insert(0, current_dir)  # 确保当前目录在搜索路径中

try:
    # 直接从当前目录导入 predict 模块
    from predict import MixingLabelPredictor
    predictor = MixingLabelPredictor()
except ImportError as e:
    st.error(f"导入MixingLabelPredictor失败: {e}")
    
    # 如果导入失败，创建一个模拟的预测器类用于演示
    class MockMixingLabelPredictor:
        def predict(self, text, lang):
            if lang == "中文":
                return "高频提升", "声音空间感", "HF001"
            else:
                return "High Frequency Enhancement", "Spatial Depth", "HF001"
    
    st.warning("使用模拟预测器替代真实分类器。请确保predict.py文件在同一目录下。")
    predictor = MockMixingLabelPredictor()

# ---------- API配置 ----------
dashscope.api_key = "sk-3b986ed51abb4ed18aadde5d41e11397"

# ---------- 界面文本配置 ----------
UI_TEXTS = {
    "中文": {
        "title": "🎚️ 混音效果智能分类系统",
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
        "footer": "© 2025 E.Stay 混音助手 | 专业音频解决方案",
        "powered_by": "基于人工智能技术",
        "input_section": "输入需求",
        "output_section": "分析结果",
        "advice_section": "混音建议",
        "examples_section": "示例输入", 
        "advice_placeholder": "点击'开始分析'生成混音建议..."
    },
    "English": {
        "title": "🎚️ AI Mixing Label Classifier",
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
        "footer": "© 2025 E.Stay Mixing Assistant | Professional Audio Solution",
        "powered_by": "Powered by AI Technology",
        "input_section": "Input Request",
        "output_section": "Analysis Results",
        "advice_section": "Mixing Advice",
        "examples_section": "Example Inputs",
        "advice_placeholder": "Click 'Analyze' to generate mixing advice..."
    }
}

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

        # 调用DashScope API
        response = Generation.call(
            model="qwq-plus-2025-03-05",
            messages=[{"role": "user", "content": prompt}],
            stream=False,  # 改为非流式响应
            result_format="message",
            temperature=0.2,
            top_p=0.7,
            max_tokens=150
        )
        
        # 解析响应
        if response.status_code == 200:
            if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                message = response.output.choices[0].get('message')
                if message and 'content' in message:
                    return message['content'].strip()
        
        return f"{UI_TEXTS[lang]['api_error']} 错误码 {response.status_code}"
                
    except Exception as e:
        return f"{UI_TEXTS[lang]['api_error']}{str(e)}"

def predict_wrapper(text, lang):
    """预测函数包装器，处理所有异常情况"""
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

# 设置页面配置
def set_page_config(lang):
    st.set_page_config(
        page_title=UI_TEXTS[lang]["title"],
        page_icon="🎚️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

# 设置自定义CSS样式
def set_custom_css():
    # 创建Streamlit风格的CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* 主要样式 */
    :root {
        --primary-color: #7c3aed;
        --primary-light: #a78bfa;
        --primary-dark: #6d28d9;
        --primary-gradient: linear-gradient(135deg, #7c3aed, #5b21b6);
        
        /* 背景颜色 */
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --card-secondary-bg: #f1f5f9;
        --header-bg: #ffffff;
        
        /* 文本颜色 */
        --text-color: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        
        /* 边框和阴影 */
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* 页面背景色 */
    .stApp {
        background-color: var(--bg-color);
    }
    
    /* 应用容器样式 */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
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
        margin-top: 5px;
        margin-bottom: 25px;
    }
    
    /* 小标题样式 */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 17px;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 20px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 4px solid var(--primary-color);
    }
    
    /* 示例按钮样式 */
    .example-button {
        background-color: var(--card-secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        color: var(--text-color);
    }
    
    .example-button:hover {
        background-color: var(--card-bg);
        border-color: var(--primary-color);
        transform: translateY(-1px);
    }
    
    /* 预测结果框样式 */
    .output-box {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: var(--shadow-sm);
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 40px;
        padding: 15px;
        border-top: 1px solid var(--border-color);
    }
    
    /* 按钮样式优化 */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.2s ease;
        height: 40px;
    }
    
    /* 主按钮样式 */
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: var(--primary-gradient);
        border: none;
    }
    
    /* 次级按钮样式 */
    .stButton > button[data-baseweb="button"]:not([kind="primary"]) {
        background-color: var(--card-secondary-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    
    /* 输入框和文本区域样式 */
    .stTextInput > div > div, .stTextArea > div > div {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
    
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
    }
    
    /* 标签样式 */
    .stTextInput label, .stTextArea label {
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 14px;
    }
    
    /* 提示文本样式 */
    .stCaption {
        color: var(--text-muted);
        font-size: 13px;
        font-style: italic;
    }
    
    /* 分隔器样式 */
    .stDivider {
        margin-top: 25px;
        margin-bottom: 25px;
    }
    
    /* 可折叠部分样式 */
    .streamlit-expanderHeader {
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary);
        border-radius: 8px;
        background-color: var(--card-secondary-bg);
    }
    
    /* 成功消息样式 */
    .stSuccess {
        background-color: #dcfce7;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 14px;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# 创建标题和副标题
def render_header(lang):
    st.markdown(f"<h1 class='app-title'>{UI_TEXTS[lang]['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='app-subtitle'>{UI_TEXTS[lang]['subtitle']}</p>", unsafe_allow_html=True)

# 创建示例部分
def render_examples(lang):
    st.markdown(f"<div class='section-title'>{UI_TEXTS[lang]['examples_section']}</div>", unsafe_allow_html=True)
    st.markdown(f"<p>{UI_TEXTS[lang]['examples_title']}</p>", unsafe_allow_html=True)
    
    # 使用列布局展示示例
    cols = st.columns(3)
    for i, example in enumerate(UI_TEXTS[lang]["examples"]):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.user_input = example
                st.session_state.run_analysis = True
                st.experimental_rerun()

# 创建输入部分
def render_input_section(lang):
    st.markdown(f"<div class='section-title'>{UI_TEXTS[lang]['input_section']}</div>", unsafe_allow_html=True)
    
    # 如果session_state中有user_input，使用它作为默认值
    default_value = st.session_state.user_input if "user_input" in st.session_state else ""
    
    user_input = st.text_area(
        label=UI_TEXTS[lang]["input_label"],
        value=default_value,
        height=100,
        placeholder=UI_TEXTS[lang]["examples"][0],
        key="input_area"
    )
    
    # 按钮行
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        analyze_clicked = st.button(
            UI_TEXTS[lang]["analyze_btn"],
            type="primary",
            use_container_width=True,
            key="analyze_button"
        )
    with col2:
        clear_clicked = st.button(
            UI_TEXTS[lang]["clear_btn"],
            type="secondary",
            use_container_width=True,
            key="clear_button"
        )
    with col3:
        lang_clicked = st.button(
            UI_TEXTS[lang]["switch_lang"],
            type="secondary",
            use_container_width=True,
            key="lang_button"
        )
        
    return user_input, analyze_clicked, clear_clicked, lang_clicked

# 创建输出部分
def render_output_section(lang, prediction_label="", prediction_code="", advice=""):
    if prediction_label or prediction_code or advice:
        st.markdown(f"<div class='section-title'>{UI_TEXTS[lang]['output_section']}</div>", unsafe_allow_html=True)
        
        # 输出框
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(UI_TEXTS[lang]["output_label"], value=prediction_label, disabled=True, key="output_label")
        with col2:
            st.text_input(UI_TEXTS[lang]["output_code"], value=prediction_code, disabled=True, key="output_code")
        
        # 混音建议部分
        st.markdown(f"<div class='section-title'>{UI_TEXTS[lang]['advice_section']}</div>", unsafe_allow_html=True)
        
        # 可编辑的混音建议
        advice_value = advice if advice else UI_TEXTS[lang]["advice_placeholder"]
        edited_advice = st.text_area(
            label="",
            value=advice_value,
            height=150,
            key="advice_area"
        )
        st.caption(UI_TEXTS[lang]["edit_hint"])
        
        # 复制按钮
        if advice:
            copy_clicked = st.button(
                UI_TEXTS[lang]["copy_btn"],
                type="secondary", 
                key="copy_button"
            )
            
            # 处理复制按钮点击
            if copy_clicked:
                st.session_state.show_copy_success = True
            
            # 显示复制成功消息
            if st.session_state.get("show_copy_success", False):
                st.success(UI_TEXTS[lang]["copy_success"])
                # 3秒后自动隐藏消息
                import time
                time.sleep(1.5)
                st.session_state.show_copy_success = False
        
        return edited_advice
    return ""

# 创建工具箱部分
def render_toolbox(lang):
    with st.expander(UI_TEXTS[lang]["toolbox_title"]):
        paste_content = st.text_area(
            UI_TEXTS[lang]["paste_btn"],
            value=st.session_state.get("paste_content", ""),
            placeholder=UI_TEXTS[lang]["paste_placeholder"],
            height=100,
            key="paste_area"
        )
        
        st.caption(UI_TEXTS[lang]["edit_hint"])
        
        # 添加一个传递到输入区的按钮
        if paste_content:
            if st.button(UI_TEXTS[lang]["input_label"], key="paste_to_input"):
                st.session_state.user_input = paste_content
                st.experimental_rerun()
    
    return paste_content

# 创建页脚
def render_footer(lang):
    st.markdown(f"""
    <div class='footer'>
        {UI_TEXTS[lang]['footer']}
        <div style="margin-top: 5px; font-size: 12px;">
            {UI_TEXTS[lang]['powered_by']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# 主应用函数
def main():
    # 初始化 session state
    if 'lang' not in st.session_state:
        st.session_state.lang = "中文"
    if 'prediction_label' not in st.session_state:
        st.session_state.prediction_label = ""
    if 'prediction_code' not in st.session_state:
        st.session_state.prediction_code = ""
    if 'advice' not in st.session_state:
        st.session_state.advice = ""
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    if 'paste_content' not in st.session_state:
        st.session_state.paste_content = ""
    if 'show_copy_success' not in st.session_state:
        st.session_state.show_copy_success = False
    
    # 设置页面配置
    set_page_config(st.session_state.lang)
    set_custom_css()
    
    # 渲染页面组件
    render_header(st.session_state.lang)
    render_examples(st.session_state.lang)
    
    # 输入区域
    user_input, analyze_clicked, clear_clicked, lang_clicked = render_input_section(st.session_state.lang)
    
    # 更新session_state中的user_input
    if user_input != st.session_state.user_input:
        st.session_state.user_input = user_input
    
    # 处理语言切换
    if lang_clicked:
        st.session_state.lang = "English" if st.session_state.lang == "中文" else "中文"
        st.experimental_rerun()
    
    # 处理清空
    if clear_clicked:
        st.session_state.user_input = ""
        st.session_state.prediction_label = ""
        st.session_state.prediction_code = ""
        st.session_state.advice = ""
        st.experimental_rerun()
    
    # 处理分析
    # 检查是否应该运行分析（从按钮点击或示例点击）
    should_analyze = (analyze_clicked and user_input) or st.session_state.run_analysis
    
    if should_analyze:
        # 重置运行标志
        st.session_state.run_analysis = False
        
        with st.spinner(UI_TEXTS[st.session_state.lang]["generating"]):
            label, code, advice = predict_wrapper(user_input, st.session_state.lang)
            st.session_state.prediction_label = label
            st.session_state.prediction_code = code
            st.session_state.advice = advice
    
    # 输出区域
    edited_advice = render_output_section(
        st.session_state.lang,
        st.session_state.prediction_label,
        st.session_state.prediction_code,
        st.session_state.advice
    )
    
    # 如果建议被编辑，更新session状态
    if edited_advice and edited_advice != st.session_state.advice and not st.session_state.advice == "":
        st.session_state.advice = edited_advice
    
    # 工具箱
    paste_content = render_toolbox(st.session_state.lang)
    if paste_content != st.session_state.paste_content:
        st.session_state.paste_content = paste_content
    
    # 页脚
    render_footer(st.session_state.lang)

# 运行应用
if __name__ == "__main__":
    main()