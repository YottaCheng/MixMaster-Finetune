import gradio as gr
from predict import MixingLabelPredictor

# 定义 UI 文本（中英文），所有静态文本均在此定义
ui_texts = {
    "中文": {
        "toggle_button": "切换语言 (当前: 中文)",
        "title": "# 混音效果标签预测系统",
        "input_label": "输入混音需求描述（中英文均可）",
        "output_label": "预测标签",
        "output_code": "标签代码",
        "description": "输入音频处理需求，系统将自动推荐标签。",
        "generate_button": "生成",
        "examples": [["人声高频需要更明亮"], ["增加鼓组的压缩感"], ["空间感不够宽广"]]
    },
    "English": {
        "toggle_button": "Toggle Language (Current: English)",
        "title": "# Mixing Effects Label Prediction System",
        "input_label": "Enter audio processing requirements (Chinese/English)",
        "output_label": "Predicted Label",
        "output_code": "Label Code",
        "description": "Enter your audio processing requirements, and the system will automatically recommend labels.",
        "generate_button": "Generate",
        "examples": [["The vocals need to be brighter"], ["Increase compression on drums"], ["The space is not wide enough"]]
    }
}

# 初始化预测器（确保模型文件在 ../data/outputs 下）
predictor = MixingLabelPredictor(model_dir="../data/outputs")

def predict_with_language(input_text, lang):
    """
    根据当前语言状态调用预测器进行预测。
    预测器的 predict 方法应定义为：
    
        def predict(self, input_text, lang="中文"):
            english_text = self.translate_to_english(input_text).lower()
            text_vector = self.vectorizer.transform([english_text])
            predicted_label = self.model.predict(text_vector)[0]
            predicted_label_zh = self.label_mapping[predicted_label][0]
            predicted_label_en = self.label_mapping[predicted_label][1]
            label_code = predicted_label
            return predicted_label_zh, predicted_label_en, label_code

    此处根据 lang 返回对应语言的预测标签和标签代码。
    """
    pred_zh, pred_en, label_code = predictor.predict(input_text, lang=lang)
    if lang == "中文":
        return pred_zh, label_code
    else:
        return pred_en, label_code

def toggle_language(current_lang):
    """切换语言状态：如果当前为中文则切换为英文，反之亦然"""
    return "English" if current_lang == "中文" else "中文"

with gr.Blocks(css="""
    /* 简单自定义 CSS，让界面更简洁美观 */
    #lang-toggle { padding: 4px 8px; font-size: 0.9rem; }
    #main-content { padding: 20px; border: 1px solid #ddd; border-radius: 8px; margin-top: 20px; }
    body { background-color: #f8f8f8; }
""") as demo:
    # 定义一个 State 组件用于存储当前语言（初始为 "中文"）
    lang_state = gr.State("中文")
    
    # 顶部区域：左侧标题，右上角放置语言切换按钮
    with gr.Row():
        title_component = gr.Markdown(ui_texts["中文"]["title"])
        lang_toggle = gr.Button(ui_texts["中文"]["toggle_button"], elem_id="lang-toggle")
    
    # 主体内容区域（卡片式设计）
    with gr.Column(elem_id="main-content"):
        description_component = gr.Markdown(ui_texts["中文"]["description"])
        # 使用 Markdown 组件显示输入框前的标签
        input_label_md = gr.Markdown(ui_texts["中文"]["input_label"])
        input_text = gr.Textbox(placeholder="在此输入您的需求……")
        generate_button = gr.Button(ui_texts["中文"]["generate_button"], variant="primary")
        # 输出区域：在输出框前分别添加 Markdown 显示标签
        with gr.Row():
            output_label_md = gr.Markdown(ui_texts["中文"]["output_label"])
            output_label = gr.Textbox()
        with gr.Row():
            output_code_md = gr.Markdown(ui_texts["中文"]["output_code"])
            output_code = gr.Textbox()
    
    # 示例区（页面底部）
    gr.Examples(
        examples=ui_texts["中文"]["examples"],
        inputs=input_text,
        cache_examples=False
    )
    
    # 事件绑定：
    # 1. 点击生成按钮时，根据当前语言调用预测函数
    generate_button.click(fn=predict_with_language, inputs=[input_text, lang_state],
                            outputs=[output_label, output_code])
    # 2. 点击语言切换按钮时切换语言状态
    lang_toggle.click(fn=toggle_language, inputs=lang_state, outputs=lang_state)
    
    # 3. 当语言状态变化时，利用 gr.update() 动态更新所有静态文本组件
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["toggle_button"]),
                      inputs=lang_state, outputs=lang_toggle)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["title"]),
                      inputs=lang_state, outputs=title_component)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["description"]),
                      inputs=lang_state, outputs=description_component)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["input_label"]),
                      inputs=lang_state, outputs=input_label_md)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["output_label"]),
                      inputs=lang_state, outputs=output_label_md)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["output_code"]),
                      inputs=lang_state, outputs=output_code_md)
    lang_state.change(fn=lambda lang: gr.update(value=ui_texts[lang]["generate_button"]),
                      inputs=lang_state, outputs=generate_button)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)

    # 添加加载动画
css = """
.spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

with gr.Blocks(css=css) as demo:
    loading = gr.HTML("<div class='spinner'></div>", visible=False)
    
    def predict_wrapper(text, lang):
        loading.update(visible=True)
        try:
            result = predict_with_language(text, lang)
        except Exception as e:
            result = ("预测失败", str(e))
        finally:
            loading.update(visible=False)
        return result
    
    generate_button.click(
        fn=predict_wrapper,
        inputs=[input_text, lang_state],
        outputs=[output_label, output_code],
        show_progress="hidden"
    )