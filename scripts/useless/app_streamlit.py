import sys
import os
import streamlit as st
import dashscope
from dashscope import Generation
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit.components.v1 as components
import json
import re
import base64

# Add current path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import predictor (assuming the predict.py file exists in the same directory)
from predict import MixingLabelPredictor

# Initialize predictor
predictor = MixingLabelPredictor()

# API configuration
dashscope.api_key = "sk-3b986ed51abb4ed18aadde5d41e11397"

# UI text configuration
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

# Define CSS styles
def get_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-container {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    
    .title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        color: #1e293b;
        position: relative;
        display: inline-block;
    }
    
    .subtitle {
        font-size: 15px;
        color: #64748b;
        font-weight: 400;
    }
    
    .section-title {
        font-size: 17px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 16px;
        color: #1e293b;
        display: flex;
        align-items: center;
        position: relative;
        padding-left: 8px;
    }
    
    .edit-hint {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        text-align: right;
        font-style: italic;
    }
    
    .example-button {
        background-color: #f1f5f9;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 14px;
        margin: 0 8px 8px 0;
        display: inline-block;
    }
    
    .example-button:hover {
        background-color: #ffffff;
        border-color: #a78bfa;
        transform: translateY(-1px);
    }
    
    .footer {
        text-align: center;
        font-size: 13px;
        color: #94a3b8;
        margin-top: 32px;
        padding: 16px;
        border-top: 1px solid #e2e8f0;
        letter-spacing: 0.02em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        color: white;
        border: none;
        box-shadow: 0 2px 5px rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(124, 58, 237, 0.5);
        filter: brightness(1.05);
    }
    
    .secondary-btn {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .secondary-btn:hover {
        background-color: #ffffff !important;
        border-color: #a78bfa !important;
        transform: translateY(-1px) !important;
    }
    
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        gap: 8px;
    }
    
    .copy-btn {
        background: #f1f5f9;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        padding: 8px 16px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    
    .copy-btn:hover {
        background-color: #ffffff;
        border-color: #a78bfa;
        transform: translateY(-1px);
    }
    
    .copy-success {
        display: none;
        margin-left: 8px;
        color: #10b981;
        font-size: 14px;
    }
    </style>
    """

# Get mixing advice using DashScope API
def get_mixing_advice(user_input, label, lang="中文"):
    """Call DashScope API to generate mixing advice"""
    try:
        # Build prompt
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

        # Call DashScope API
        response = Generation.call(
            model="qwq-plus-2025-03-05",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=150
        )
        
        if response.status_code == 200:
            if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                message = response.output.choices[0].get('message')
                if message and 'content' in message:
                    return message['content'].strip()
        
        return f"{UI_TEXTS[lang]['api_error']} 错误码 {response.status_code}"
            
    except Exception as e:
        return f"{UI_TEXTS[lang]['api_error']}{str(e)}"

# Predict wrapper
def predict_wrapper(text, lang):
    try:
        # Get prediction, always get both languages' results
        zh_main, zh_secondary, code = predictor.predict(text, "中文")
        en_main, en_secondary, _ = predictor.predict(text, "English")
        
        # Get mixing advice using the label in current language
        advice = get_mixing_advice(
            text, 
            zh_main if lang == "中文" else en_main, 
            lang
        )
        
        # Format label based on language
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

# Create example buttons using HTML
def create_examples_html(lang):
    examples_html = ""
    for i, example in enumerate(UI_TEXTS[lang]["examples"]):
        # Use data attribute instead of inline onclick
        examples_html += f'<div class="example-button" data-example-id="{i}">{example}</div>'
    
    js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const exampleButtons = document.querySelectorAll('.example-button');
        exampleButtons.forEach(button => {
            button.addEventListener('click', function() {
                const exampleId = this.getAttribute('data-example-id');
                const examples = %s;
                const text = examples[exampleId];
                
                const textareaElement = document.querySelector('textarea');
                if (textareaElement) {
                    textareaElement.value = text;
                    // Trigger change event
                    const event = new Event('input', { bubbles: true });
                    textareaElement.dispatchEvent(event);
                    
                    // Find analyze button and click it
                    setTimeout(() => {
                        const analyzeButton = Array.from(document.querySelectorAll('button')).find(
                            button => button.innerText.includes('Analyze') || button.innerText.includes('开始分析')
                        );
                        if (analyzeButton) analyzeButton.click();
                    }, 100);
                }
            });
        });
    });
    </script>
    """ % json.dumps(UI_TEXTS[lang]["examples"])
    
    return examples_html + js

# Add a copy-to-clipboard functionality without using f-strings
def add_copy_button_html(lang, button_id="copy-button"):
    # Create a div with a button that calls a JS function to copy text
    html = f'''
    <div style="margin-top: 10px;">
        <button id="{button_id}" class="copy-btn">{UI_TEXTS[lang]["copy_btn"]}</button>
        <span id="copy-success" class="copy-success">{UI_TEXTS[lang]["copy_success"]}</span>
    </div>
    <script>
    document.getElementById("{button_id}").addEventListener("click", function() {{
        // Get the content from textarea
        var textarea = document.querySelector('[data-testid="stTextArea"] textarea');
        if (textarea) {{
            // Copy the content to clipboard
            navigator.clipboard.writeText(textarea.value).then(function() {{
                // Show success message
                document.getElementById("copy-success").style.display = "inline";
                setTimeout(function() {{
                    document.getElementById("copy-success").style.display = "none";
                }}, 2000);
            }});
        }}
    }});
    </script>
    '''
    return html

# Main Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="AI Mixing Label Classifier", 
        page_icon="🎚️", 
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Apply CSS
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Initialize session state if not exists
    if 'lang' not in st.session_state:
        st.session_state.lang = "中文"
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    if 'output_label' not in st.session_state:
        st.session_state.output_label = ""
    
    if 'output_code' not in st.session_state:
        st.session_state.output_code = ""
    
    if 'output_advice' not in st.session_state:
        st.session_state.output_advice = ""
    
    # Current language
    lang = st.session_state.lang
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and subtitle
    st.markdown(f'<div class="title">{UI_TEXTS[lang]["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{UI_TEXTS[lang]["subtitle"]}</div>', unsafe_allow_html=True)
    
    # Language switch in the sidebar
    with st.sidebar:
        if st.button(UI_TEXTS[lang]["switch_lang"]):
            st.session_state.lang = "English" if lang == "中文" else "中文"
            st.experimental_rerun()
    
    # Input section
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["input_section"]}</div>', unsafe_allow_html=True)
    
    # Input text area
    user_input = st.text_area(
        UI_TEXTS[lang]["input_label"],
        value=st.session_state.user_input,
        height=100,
        placeholder=UI_TEXTS[lang]["examples"][0]
    )
    
    # Store input in session state
    st.session_state.user_input = user_input
    
    # Button row
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        analyze_clicked = st.button(UI_TEXTS[lang]["analyze_btn"], use_container_width=True)
    
    with col2:
        if st.button(UI_TEXTS[lang]["clear_btn"], use_container_width=True):
            st.session_state.user_input = ""
            st.session_state.output_label = ""
            st.session_state.output_code = ""
            st.session_state.output_advice = ""
            st.experimental_rerun()
    
    # Process when analyze button is clicked
    if analyze_clicked and user_input:
        with st.spinner(UI_TEXTS[lang]["generating"]):
            label, code, advice = predict_wrapper(user_input, lang)
            
            # Store results in session state
            st.session_state.output_label = label
            st.session_state.output_code = code
            st.session_state.output_advice = advice
    
    # Output section
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["output_section"]}</div>', unsafe_allow_html=True)
    
    # Display output in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(UI_TEXTS[lang]["output_label"], value=st.session_state.output_label, disabled=True)
    
    with col2:
        st.text_input(UI_TEXTS[lang]["output_code"], value=st.session_state.output_code, disabled=True)
    
    # Advice section
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["advice_section"]}</div>', unsafe_allow_html=True)
    
    # Editable advice area
    advice_value = st.text_area(
        "",
        value=st.session_state.output_advice,
        height=150,
        placeholder=UI_TEXTS[lang]["advice_placeholder"],
        key="advice_area"
    )
    
    # Update advice in session state
    st.session_state.output_advice = advice_value
    
    # Edit hint and copy button
    st.markdown(f'<div class="edit-hint">{UI_TEXTS[lang]["edit_hint"]}</div>', unsafe_allow_html=True)
    
    # Copy button for advice
    if st.session_state.output_advice:
        st.markdown(add_copy_button_html(lang), unsafe_allow_html=True)
    
    # Examples section
    st.markdown(f'<div class="section-title">{UI_TEXTS[lang]["examples_section"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p>{UI_TEXTS[lang]["examples_title"]}</p>', unsafe_allow_html=True)
    
    # Examples display
    st.components.v1.html(
        create_examples_html(lang),
        height=80,
        scrolling=False
    )
    
    # Toolbox (collapsible section)
    with st.expander(UI_TEXTS[lang]["toolbox_title"]):
        paste_text = st.text_area(
            UI_TEXTS[lang]["paste_btn"],
            height=100,
            placeholder=UI_TEXTS[lang]["paste_placeholder"]
        )
        
        if paste_text:
            if st.button(UI_TEXTS[lang]["paste_btn"]):
                st.session_state.user_input = paste_text
                st.experimental_rerun()
    
    # Footer
    st.markdown(
        f"""
        <div class="footer">
            {UI_TEXTS[lang]["footer"]}
            <div style="margin-top: 5px; font-size: 12px;">
                {UI_TEXTS[lang]["powered_by"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()