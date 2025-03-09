import sys
import os

# ---------- 关键修复：必须在导入前添加路径 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 使用 insert(0) 确保优先搜索当前目录

import gradio as gr
from predict import MixingLabelPredictor

# 初始化预测器
predictor = MixingLabelPredictor()

# ---------- 界面文本配置 ----------
UI_TEXTS = {
    "中文": {
        "title": "🎚️ 混音效果智能分类系统",
        "input_label": "请输入音频处理需求（中英文均可）",
        "output_label": "预测标签",
        "output_code": "标签代码",
        "examples": [
            ["人声高频需要更明亮"],
            ["增加鼓组的冲击力"],
            ["整体空间感不足"]
        ],
        "error_msg": "⚠️ 预测失败："
    },
    "English": {
        "title": "🎚️ AI Mixing Label Classifier",
        "input_label": "Enter audio processing request (Chinese/English)",
        "output_label": "Predicted Label",
        "output_code": "Label Code",
        "examples": [
            ["Vocals need more brightness"],
            ["Increase drum punchiness"],
            ["Lack of overall spatial depth"]
        ],
        "error_msg": "⚠️ Prediction failed:"
    }
}

# ---------- 核心逻辑 ----------
def predict_wrapper(text, lang):
    """带错误处理的预测函数"""
    try:
        zh_label, en_label, code = predictor.predict(text)
        if lang == "中文":
            return zh_label, code
        return en_label, code
    except Exception as e:
        error_msg = f"{UI_TEXTS[lang]['error_msg']}{str(e)}"
        return error_msg, "ERROR"

def toggle_language(current_lang):
    """语言切换"""
    return "English" if current_lang == "中文" else "中文"

# ---------- 界面构建 ----------
css = """
#main-block {
    background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.output-box {
    background-color: #333333 !important;
    border: 1px solid #4a4a4a !important;
    color: #ffffff !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="purple")) as app:
    # 状态存储
    lang_state = gr.State("中文")
    
    # 主界面布局
    with gr.Column(elem_id="main-block"):
        # 标题栏
        gr.Markdown("## " + UI_TEXTS["中文"]["title"])
        
        # 输入区
        input_box = gr.Textbox(
            label=UI_TEXTS["中文"]["input_label"],
            placeholder="例如：'人声不够清晰'...",
            lines=2
        )
        
        # 功能按钮
        with gr.Row():
            submit_btn = gr.Button("🚀 开始分析", variant="primary")
            lang_btn = gr.Button("🌐 切换语言", variant="secondary")

        # 输出区
        with gr.Column():
            gr.Markdown("### 分析结果")
            output_label = gr.Textbox(label=UI_TEXTS["中文"]["output_label"], elem_classes="output-box")
            output_code = gr.Textbox(label=UI_TEXTS["中文"]["output_code"], elem_classes="output-box")

        # 示例区
        gr.Examples(
            examples=UI_TEXTS["中文"]["examples"],
            inputs=input_box,
            label="💡 示例输入"
        )

    # ---------- 事件绑定 ----------
    # 提交按钮点击事件
    submit_btn.click(
        fn=predict_wrapper,
        inputs=[input_box, lang_state],
        outputs=[output_label, output_code]
    )

    # 语言切换事件
    lang_btn.click(
        fn=toggle_language,
        inputs=lang_state,
        outputs=lang_state
    )
    
    # 动态更新文本
    lang_state.change(
        lambda lang: {
            input_box: gr.update(label=UI_TEXTS[lang]["input_label"]),
            output_label: gr.update(label=UI_TEXTS[lang]["output_label"]),
            output_code: gr.update(label=UI_TEXTS[lang]["output_code"]),
            submit_btn: gr.update(value="🚀 Analyze" if lang == "English" else "🚀 开始分析"),
            lang_btn: gr.update(value="🌐 Switch Language" if lang == "English" else "🌐 切换语言")
        },
        inputs=lang_state,
        outputs=[input_box, output_label, output_code, submit_btn, lang_btn]
    )

# ---------- 启动服务 ----------
if __name__ == "__main__":
    app.launch(
        server_port=7860,
        share=True,
        favicon_path="https://example.com/favicon.ico"  # 可替换本地图标路径
    )