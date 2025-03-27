import sys
import os
import dashscope
from dashscope import Generation

# ---------- 关键修复：必须在导入前添加路径 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 使用 insert(0) 确保优先搜索当前目录

import gradio as gr
from predict import MixingLabelPredictor

# 初始化预测器
predictor = MixingLabelPredictor()

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
        "switch_lang": "🌐 switch to Eng",  # 修改为更直观的描述
        "examples_title": "💡 示例输入",
        "examples": [
            ["人声高频需要更明亮"],
            ["增加鼓组的冲击力"],
            ["整体空间感不足"]
        ],
        "error_msg": "⚠️ 预测失败：",
        "generating": "正在生成混音建议...",
        "api_error": "⚠️ 生成建议失败：",
        "paste_toolbox": "复制/粘贴工具箱",
        "toolbox_title": "工具箱",  # 简化标题
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
        "switch_lang": "🌐 Switch to Chinese",  # 修改为更直观的描述
        "examples_title": "💡 Example Inputs",
        "examples": [
            ["Vocals need more brightness"],
            ["Increase drum punchiness"],
            ["Lack of overall spatial depth"]
        ],
        "error_msg": "⚠️ Prediction failed: ",
        "generating": "Generating mixing advice...",
        "api_error": "⚠️ Failed to generate advice: ",
        "paste_toolbox": "Copy/Paste Toolbox",
        "toolbox_title": "Toolbox",  # 简化标题
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
            stream=True,
            incremental_output=True,
            result_format="message",
            temperature=0.2,
            top_p=0.7,
            max_tokens=150
        )
        
        # 解析流式响应
        full_response = []
        for chunk in response:
            if chunk.status_code == 200:
                if hasattr(chunk.output, 'choices') and len(chunk.output.choices) > 0:
                    message = chunk.output.choices[0].get('message')
                    if message and 'content' in message:
                        full_response.append(message['content'])
            else:
                return f"{UI_TEXTS[lang]['api_error']} 错误码 {chunk.status_code}"
                
        return ''.join(full_response).strip()
            
    except Exception as e:
        return f"{UI_TEXTS[lang]['api_error']}{str(e)}"

# 修改后的predict_wrapper
def predict_wrapper(text, lang):
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

def toggle_language(current_lang):
    """语言切换"""
    return "English" if current_lang == "中文" else "中文"

def clear_input():
    """清空输入"""
    return ""

def copy_notification(text, lang):
    """显示复制成功通知"""
    return UI_TEXTS[lang]["copy_success"]

# ---------- 示例生成函数 ----------
def create_examples_html(lang):
    examples_html = ""
    for example in UI_TEXTS[lang]["examples"]:
        examples_html += f"""
        <div class="example-item" 
            onclick="document.querySelector('#input-box textarea').value = '{example[0]}';
            document.querySelector('.primary-btn').click();">
            {example[0]}
        </div>
        """
    return f"""
    <div class="examples-container" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        {examples_html}
    </div>
    """

# ---------- 更新标签的函数 ----------
def update_labels(lang):
    """针对旧版Gradio，创建新的组件实例来更新标签"""
    # 这是关键函数，适用于旧版Gradio无法直接更新label属性的情况
    
    # 更新输入框
    new_input = gr.Textbox(
        label=UI_TEXTS[lang]["input_label"],
        value="",  # 保持输入框内容为空
        placeholder=UI_TEXTS[lang]["examples"][0][0],
        lines=3
    )
    
    # 更新输出标签
    new_output_label = gr.Textbox(
        label=UI_TEXTS[lang]["output_label"],
        value="",  # 保持内容为空
        elem_classes="output-box"
    )
    
    # 更新输出代码
    new_output_code = gr.Textbox(
        label=UI_TEXTS[lang]["output_code"],
        value="",  # 保持内容为空
        elem_classes="output-box"
    )
    
    # 更新粘贴区域
    new_paste_area = gr.Textbox(
        label=UI_TEXTS[lang]["paste_btn"],
        value="",  # 保持内容为空
        placeholder=UI_TEXTS[lang]["paste_placeholder"],
        lines=2
    )
    
    # 返回创建的新组件
    return new_input, new_output_label, new_output_code, new_paste_area

# ---------- CSS样式 ----------
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* 修改后的亮色主题CSS变量 */
:root {
    /* 主色调 - 紫色渐变系列 */
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
    
    /* 输入和交互区域 */
    --input-bg: #ffffff;
    --input-border: #cbd5e1;
    --input-focus: #7c3aed;
    
    /* 状态颜色 */
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;

    /* 圆角和间距 */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
}

/* 基础样式 */
body {
    background-color: var(--bg-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.5;
}

/* 主容器样式 */
#main-block {
    background: var(--card-bg);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
    margin-bottom: var(--spacing-lg);
    position: relative;
    overflow: hidden;
}

/* 背景装饰效果 */
#main-block::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    height: 200px;
    width: 200px;
    background: radial-gradient(circle at top right, rgba(124, 58, 237, 0.05), transparent 70%);
    z-index: 0;
    pointer-events: none;
}

/* 标题区样式 */
.header {
    background: var(--header-bg);
    margin: calc(-1 * var(--spacing-lg));
    margin-bottom: var(--spacing-lg);
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
}

.header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary-color), transparent);
}

.title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: var(--spacing-xs);
    color: var(--text-color);
    position: relative;
    display: inline-block;
}

.title::after {
    content: "";
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 40px;
    height: 3px;
    background: var(--primary-gradient);
    border-radius: 10px;
}

.subtitle {
    font-size: 15px;
    color: var(--text-secondary);
    font-weight: 400;
}

/* 区域标题样式 */
.section-title {
    font-size: 17px;
    font-weight: 600;
    margin-top: var(--spacing-lg);
    margin-bottom: var(--spacing-md);
    color: var(--text-color);
    display: flex;
    align-items: center;
    position: relative;
    padding-left: var(--spacing-sm);
}

.section-title::before {
    content: "";
    position: absolute;
    left: 0;
    width: 4px;
    height: 18px;
    background: var(--primary-gradient);
    border-radius: var(--radius-sm);
}

/* 输入和输出区域样式 */
.input-area textarea, .output-area textarea {
    border: 1px solid var(--input-border) !important;
    border-radius: var(--radius-md) !important;
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    padding: var(--spacing-md) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    font-size: 15px !important;
    box-shadow: var(--shadow-sm) !important;
}

.input-area textarea:focus, .output-area textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
    outline: none !important;
}

.output-box textarea {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--text-color) !important;
    border-radius: var(--radius-md) !important;
    font-size: 15px !important;
}

.advice-box textarea {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--text-color) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
    min-height: 120px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* 按钮样式 */
.button-row {
    display: flex;
    gap: var(--spacing-sm);
    margin: var(--spacing-md) 0;
}

button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
    font-size: 14px !important;
    height: 40px !important;
    min-width: 100px !important;
}

.primary-btn {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 2px 5px rgba(124, 58, 237, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(124, 58, 237, 0.5) !important;
    filter: brightness(1.05) !important;
}

.secondary-btn {
    background-color: var(--card-secondary-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

.secondary-btn:hover {
    background-color: var(--card-bg) !important;
    border-color: var(--primary-light) !important;
    transform: translateY(-1px) !important;
}

/* 通知和提示样式 */
.edit-hint {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: var(--spacing-xs);
    text-align: right;
    font-style: italic;
}

.success-message {
    font-size: 14px;
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: var(--success-color);
    color: white;
    border-radius: var(--radius-md);
    display: inline-block;
    margin-left: var(--spacing-sm);
    animation: fadeIn 0.3s ease-out;
}

/* 页脚样式 */
.footer {
    text-align: center;
    font-size: 13px;
    color: var(--text-muted);
    margin-top: var(--spacing-xl);
    padding: var(--spacing-md);
    border-top: 1px solid var(--border-color);
    letter-spacing: 0.02em;
}

/* 标签样式 */
label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin-bottom: var(--spacing-xs) !important;
    letter-spacing: 0.01em !important;
}

/* 示例样式 */
.gr-samples-table {
    background: transparent !important;
    border: none !important;
}

.gr-samples-table td {
    background-color: var(--card-secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s !important;
}

.gr-samples-table td:hover {
    background-color: var(--card-bg) !important;
    border-color: var(--primary-light) !important;
    cursor: pointer !important;
}

/* 动画效果 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(124, 58, 237, 0); }
    100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
}

/* 滚动条样式 */
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

/* 响应式调整 */
@media (max-width: 768px) {
    .header, #main-block {
        padding: 16px;
    }
    
    .title {
        font-size: 22px;
    }
    
    .subtitle {
        font-size: 14px;
    }
}

/* 示例项目样式 */
.example-item {
    background-color: var(--card-secondary-bg);
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
}

.example-item:hover {
    background-color: var(--card-bg);
    border-color: var(--primary-light);
    transform: translateY(-1px);
}
"""

# ---------- 主界面构建 ----------
with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="purple")) as app:
    # ===== 创建隐藏状态 =====
    lang_state = gr.State("中文")
    
    # ===== 主界面布局 =====
    with gr.Column(elem_id="main-block"):
        # ==== 标题区 ====
        with gr.Column(elem_classes="header"):
            # 使用直接初始化方式
            title_md = gr.Markdown(f"<div class='title'>{UI_TEXTS['中文']['title']}</div>")
            subtitle_md = gr.Markdown(f"<div class='subtitle'>{UI_TEXTS['中文']['subtitle']}</div>")

        # ==== 输入区 ====
        # 直接初始化所有Markdown组件
        input_section_md = gr.Markdown(f"<div class='section-title'>{UI_TEXTS['中文']['input_section']}</div>")
        
        input_box = gr.Textbox(
            label=UI_TEXTS["中文"]["input_label"],
            placeholder=UI_TEXTS["中文"]["examples"][0][0],
            lines=3,
            elem_id="input-box"
        )

        # ==== 功能按钮 ====
        with gr.Row(elem_classes="button-row"):
            submit_btn = gr.Button(UI_TEXTS["中文"]["analyze_btn"], elem_classes="primary-btn", elem_id="submit-btn")
            clear_btn = gr.Button(UI_TEXTS["中文"]["clear_btn"], elem_classes="secondary-btn")
            lang_btn = gr.Button(UI_TEXTS["中文"]["switch_lang"], elem_classes="secondary-btn")
            copy_advice_btn = gr.Button(UI_TEXTS["中文"]["copy_btn"], elem_classes="secondary-btn")

        # ==== 输出区 ====
        output_section_md = gr.Markdown(f"<div class='section-title'>{UI_TEXTS['中文']['output_section']}</div>")
        
        with gr.Row():
            output_label = gr.Textbox(label=UI_TEXTS["中文"]["output_label"], elem_classes="output-box")
            output_code = gr.Textbox(label=UI_TEXTS["中文"]["output_code"], elem_classes="output-box")

        # ==== 混音建议 ====
        advice_section_md = gr.Markdown(f"<div class='section-title'>{UI_TEXTS['中文']['advice_section']}</div>")
        
        output_advice = gr.Textbox(
            label="",
            elem_classes="advice-box",
            lines=4,
            placeholder=UI_TEXTS["中文"]["advice_placeholder"],
            interactive=True
        )
        edit_hint_md = gr.Markdown(f"<div class='edit-hint'>{UI_TEXTS['中文']['edit_hint']}</div>")
        
        # 复制状态显示
        copy_status = gr.Markdown("")

        # ==== 示例区 ====
        examples_section_md = gr.Markdown(f"<div class='section-title'>{UI_TEXTS['中文']['examples_section']}</div>")
        examples_title_md = gr.Markdown(f"<p>{UI_TEXTS['中文']['examples_title']}</p>")
        
        # 使用动态HTML显示示例
        examples_display = gr.HTML(create_examples_html("中文"), elem_id="examples-display")

        # ==== 复制/粘贴工具 ====
        # 使用空标题Accordion，标题放在内部
        with gr.Accordion(open=False) as toolbox_acc:
            # 添加内部标题组件
            toolbox_title = gr.Markdown(UI_TEXTS["中文"]["toolbox_title"])
            
            paste_area = gr.Textbox(
                label=UI_TEXTS["中文"]["paste_btn"],
                placeholder=UI_TEXTS["中文"]["paste_placeholder"],
                lines=2
            )
            paste_hint_md = gr.Markdown(f"""<div class='edit-hint'>{UI_TEXTS['中文']['edit_hint']}</div>""")

        # ==== 页脚 ====
        footer = gr.HTML(
            value=f"""
            <div class='footer'>
                {UI_TEXTS['中文']['footer']}
                <div style="margin-top: 5px; font-size: 12px;">
                    {UI_TEXTS['中文']['powered_by']}
                </div>
            </div>
            """,
            elem_id="footer-component"
        )

    # ---------- 事件绑定 ----------
    # 提交按钮点击事件
    submit_btn.click(
        fn=predict_wrapper,
        inputs=[input_box, lang_state],
        outputs=[output_label, output_code, output_advice]
    )

    # 重要的更改：将标签更新拆分为两个阶段
    # 第一阶段：更新标准组件（Markdown、HTML等）
    lang_btn.click(
        fn=toggle_language,
        inputs=lang_state,
        outputs=lang_state
    ).then(
        fn=lambda lang: [
            # 1. 标题和副标题
            f"<div class='title'>{UI_TEXTS[lang]['title']}</div>",
            f"<div class='subtitle'>{UI_TEXTS[lang]['subtitle']}</div>",
            
            # 2. 所有区域标题
            f"<div class='section-title'>{UI_TEXTS[lang]['input_section']}</div>",
            f"<div class='section-title'>{UI_TEXTS[lang]['output_section']}</div>",
            f"<div class='section-title'>{UI_TEXTS[lang]['advice_section']}</div>",
            f"<div class='section-title'>{UI_TEXTS[lang]['examples_section']}</div>",
            
            # 5. 按钮文本
            UI_TEXTS[lang]["analyze_btn"],
            UI_TEXTS[lang]["clear_btn"],
            UI_TEXTS[lang]["switch_lang"],
            UI_TEXTS[lang]["copy_btn"],
            
            # 6. 提示和辅助文本
            f"<div class='edit-hint'>{UI_TEXTS[lang]['edit_hint']}</div>",
            f"<p>{UI_TEXTS[lang]['examples_title']}</p>",
            
            # 7. 示例区域
            create_examples_html(lang),
            
            # 9. 建议区域占位符
            UI_TEXTS[lang]["advice_placeholder"],
            
            # 10. 页脚
            f"""
            <div class='footer'>
                {UI_TEXTS[lang]['footer']}
                <div style="margin-top: 5px; font-size: 12px;">
                    {UI_TEXTS[lang]['powered_by']}
                </div>
            </div>
            """,
            
            # 11. 工具箱内部标题
            UI_TEXTS[lang]["toolbox_title"]
        ],
        inputs=[lang_state],
        outputs=[
            title_md,
            subtitle_md,
            input_section_md,
            output_section_md,
            advice_section_md,
            examples_section_md,
            submit_btn,
            clear_btn,
            lang_btn,
            copy_advice_btn,
            edit_hint_md,
            examples_title_md,
            examples_display,
            output_advice,
            footer,
            toolbox_title
        ]
    ).then(
        # 第二阶段：直接替换表单控件 - 彻底解决标签更新问题
        fn=update_labels,
        inputs=[lang_state],
        outputs=[input_box, output_label, output_code, paste_area]
    )
    
    # 复制按钮事件
    copy_advice_btn.click(
        fn=lambda advice: advice,  
        inputs=[output_advice],
        outputs=[paste_area]  
    ).then(
        fn=copy_notification, 
        inputs=[output_advice, lang_state],
        outputs=[copy_status]
    )
    
    # 清空按钮事件
    clear_btn.click(
        fn=clear_input,
        inputs=[],
        outputs=[input_box]
    )
    
    # 将粘贴区域的内容传递到输入框
    paste_area.change(
        fn=lambda text: text,
        inputs=[paste_area],
        outputs=[input_box]
    )

# ---------- 启动服务 ----------
if __name__ == "__main__":
    app.launch(
        server_port=7860,
        share=True
    )