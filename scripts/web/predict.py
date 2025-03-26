import os
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from deep_translator import GoogleTranslator
from peft import PeftModel

# 模板类，用于格式化提示词
class Template:
    """Prompt template processor"""
    
    def __init__(self, template_type="deepseek"):
        self.template_type = template_type
        
        if template_type == "deepseek":
            # Deepseek model template format
            self.system_prefix = "<|begin_of_system|>\n"
            self.system_suffix = "\n<|end_of_system|>"
            self.user_prefix = "\n<|begin_of_human|>\n"
            self.user_suffix = "\n<|end_of_human|>"
            self.assistant_prefix = "\n<|begin_of_assistant|>\n"
            self.assistant_suffix = "\n<|end_of_assistant|>"
        else:
            # ChatML template format
            self.system_prefix = "<|im_start|>system\n"
            self.system_suffix = "<|im_end|>"
            self.user_prefix = "\n<|im_start|>user\n"
            self.user_suffix = "<|im_end|>"
            self.assistant_prefix = "\n<|im_start|>assistant\n"
            self.assistant_suffix = "<|im_end|>"
    
    def encode_oneturn(self, tokenizer, messages, system=None):
        """Process single-turn dialogue and encode as token sequence"""
        # Build system prompt
        prompt = ""
        if system:
            prompt += self.system_prefix + system + self.system_suffix
        
        # Process user message
        user_content = messages[0]["content"] if len(messages) > 0 else ""
        prompt += self.user_prefix + user_content + self.user_suffix
        
        # Add assistant prefix
        prompt += self.assistant_prefix
        
        # Encode as tokens
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        return inputs, len(inputs.input_ids[0])
    
    def get_stop_token_ids(self, tokenizer):
        """Get stop token IDs"""
        stop_words = []
        if self.template_type == "deepseek":
            stop_words = ["<|end_of_assistant|>", "<|end_of_human|>"]
        elif self.template_type == "chatml":
            stop_words = ["<|im_end|>"]
        
        stop_token_ids = []
        for word in stop_words:
            try:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1:
                    stop_token_ids.append(token_ids[0])
            except:
                pass
        
        return stop_token_ids if stop_token_ids else [tokenizer.eos_token_id]

class MixingLabelPredictor:
    def __init__(self, model_dir=r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v6"):
        # 初始化检查
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ 模型目录不存在: {model_dir}")

        # 标签配置
        self.label_order = [
            "low_freq",    # 低频
            "mid_freq",     # 中频
            "high_freq",    # 高频
            "reverb",      # 混响
            "effect",      # 效果器 - 注意是单数形式
            "soundstage",   # 声场
            "compression",  # 压缩
            "volume"        # 音量
        ]
        
        # 标签映射
        self.label_mapping = self._load_label_mapping(
            os.path.join(model_dir, "music_master.json"))
        
        # 通过模板初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.template = Template("deepseek")  # 明确指定模板类型
        
        # 模型初始化标志
        self.initialized = False
        
        # 检查所有可能的LoRA适配器路径
        possible_lora_paths = [
            os.path.join(model_dir, "adapter_model"),
            os.path.join(model_dir, "lora_adapters"),
            model_dir,  # 模型目录本身可能也包含适配器文件
        ]
        
        has_lora = False
        lora_path = None
        
        # 检查哪个路径存在适配器文件
        for path in possible_lora_paths:
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                has_lora = True
                lora_path = path
                print(f"✓ 检测到LoRA适配器: {path}")
                break
        
        try:
            # 加载基础模型
            print(f"正在加载基础模型: {model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器（如果有）
            if has_lora:
                print(f"正在加载LoRA适配器: {lora_path}")
                try:
                    # 尝试加载LoRA适配器
                    self.model = PeftModel.from_pretrained(
                        self.model, 
                        lora_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto"
                    )
                    print(f"✅ LoRA适配器加载成功")
                    
                    # 合并权重以提高性能
                    print(f"正在合并LoRA权重...")
                    original_dtype = self.model.dtype
                    self.model = self.model.merge_and_unload()
                    
                    # 确保dtype保持一致
                    if self.model.dtype != original_dtype:
                        print(f"调整模型dtype: {self.model.dtype} -> {original_dtype}")
                        self.model = self.model.to(dtype=original_dtype)
                    
                    print(f"✅ LoRA权重合并成功")
                except Exception as lora_e:
                    print(f"⚠️ LoRA适配器加载失败: {str(lora_e)}，将使用基础模型")
                    # 如果加载失败，尝试不同的路径
                    for alt_path in possible_lora_paths:
                        if alt_path != lora_path and os.path.exists(os.path.join(alt_path, "adapter_config.json")):
                            try:
                                print(f"尝试备用路径: {alt_path}")
                                self.model = PeftModel.from_pretrained(
                                    self.model, 
                                    alt_path,
                                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                                )
                                self.model = self.model.merge_and_unload()
                                print(f"✅ 备用路径LoRA适配器加载成功: {alt_path}")
                                break
                            except Exception as alt_e:
                                print(f"备用路径也失败: {str(alt_e)}")
            else:
                print("未检测到LoRA适配器，使用基础模型")
            
            # 加载tokenizer
            print(f"正在加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                padding_side="left",  # 生成任务使用左padding
                truncation_side="left"
            )
            
            # 确保padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.initialized = True
            print(f"✅ 模型加载成功: {model_dir}")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}")
        
        # 翻译器
        self.translator = GoogleTranslator(source='zh-CN', target='en')
    
    def _load_label_mapping(self, path):
        """加载与训练一致的标签映射"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
            
            # 验证标签顺序与模型输出层一致
            required_labels = self.label_order
            label_mapping = {}
            for label in required_labels:
                if label not in raw_mapping:
                    raise KeyError(f"❌ 缺失必需标签: {label}")
                label_mapping[label] = {
                    "zh": raw_mapping[label][0],
                    "en": raw_mapping[label][1]
                }
            return label_mapping
        except Exception as e:
            print(f"❌ 加载标签映射失败: {str(e)}")
            raise
    
    def _contains_chinese(self, text):
        """检测文本是否包含中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def _translate_text(self, text):
        """翻译文本"""
        try:
            if self._contains_chinese(text):
                return GoogleTranslator(source='zh-CN', target='en').translate(text).lower()
            return text.lower()
        except Exception as e:
            print(f"⚠️ 翻译失败，使用原文: {str(e)}")
            return text.lower()
    
    def _generate_response(self, input_text, system_message=None, temperature=0.75):  # 使用0.75的温度参数
        """使用LLM生成回答"""
        if not hasattr(self, "initialized") or not self.initialized:
            return "Model not initialized"
        
        try:
            # 创建消息
            message = [{"role": "user", "content": input_text}]
            
            # 编码消息
            inputs, prompt_length = self.template.encode_oneturn(
                self.tokenizer, message, system_message
            )
            inputs = inputs.to(self.device)
            
            # 与LlamaFactory一致的生成配置
            generation_config = GenerationConfig(
                temperature=0.01,  # 使用0.75的温度
                top_p=0.9,
                max_new_tokens=100,  # 增加token数量
                repetition_penalty=1.2,
                do_sample=True,  # 当温度>0时启用采样
                eos_token_id=self.template.get_stop_token_ids(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # 生成回答
            with torch.no_grad():
                generate_output = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config
                )
            
            # 解码回答
            response_ids = generate_output[0, prompt_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return f"生成错误: {str(e)}"
    
    def _parse_labels(self, text):
        """解析标签"""
        # 打印原始回答以便调试
        print(f"原始响应: {text}")
        
        if not text:
            return []
        
        try:
            # 提取输出部分
            if "Output:" in text:
                text = text.split("Output:")[1].strip()
            elif "output:" in text:
                text = text.split("output:")[1].strip()
            elif "输出：" in text:
                text = text.split("输出：")[1].strip()
            
            # 清理文本
            text = re.sub(r'^[\s:"：]+', '', text)
            text = re.sub(r'[\n\r]+', ' ', text)
            
            print(f"清理后的输出: {text}")
            
            # 按分隔符分割 - 使用更宽松的分隔符匹配
            raw_labels = [label.strip() for label in re.split(r'[,，、;/\s]+', text) if label.strip()]
            print(f"分割后的标签: {raw_labels}")
            
            # 映射到标准标签
            valid_labels = []
            
            for label in raw_labels:
                label = label.lower().strip()
                # 精确匹配
                matched = False
                for code, mapping in self.label_mapping.items():
                    zh_label = mapping["zh"].lower()
                    en_label = mapping["en"].lower()
                    
                    if label == zh_label or label == en_label or label == code.lower():
                        valid_labels.append(code)
                        matched = True
                        break
                
                # 如果没有精确匹配，尝试模糊匹配
                if not matched:
                    if 'low' in label or '低' in label:
                        if 'low_freq' not in valid_labels:
                            valid_labels.append('low_freq')
                    elif 'mid' in label or '中' in label:
                        if 'mid_freq' not in valid_labels:
                            valid_labels.append('mid_freq')
                    elif 'high' in label or '高' in label:
                        if 'high_freq' not in valid_labels:
                            valid_labels.append('high_freq')
                    elif 'reverb' in label or '混响' in label:
                        if 'reverb' not in valid_labels:
                            valid_labels.append('reverb')
                    elif 'effect' in label or '效果' in label:
                        if 'effect' not in valid_labels:
                            valid_labels.append('effect')
                    elif 'stage' in label or 'field' in label or '场' in label:
                        if 'soundstage' not in valid_labels:
                            valid_labels.append('soundstage')
                    elif 'compress' in label or '压' in label:
                        if 'compression' not in valid_labels:
                            valid_labels.append('compression')
                    elif 'volume' in label or '音量' in label:
                        if 'volume' not in valid_labels:
                            valid_labels.append('volume')
            
            # 确保至少有一个标签
            if not valid_labels and raw_labels:
                # 使用更强的关键词匹配
                for label in raw_labels:
                    label = label.lower()
                    if any(x in label for x in ['low', 'bass', '低', '低频']):
                        valid_labels.append('low_freq')
                        break
                    elif any(x in label for x in ['mid', 'middle', '中', '中频']):
                        valid_labels.append('mid_freq')
                        break
                    elif any(x in label for x in ['high', 'treble', '高', '高频']):
                        valid_labels.append('high_freq')
                        break
                    elif any(x in label for x in ['reverb', 'echo', '混响', '回声']):
                        valid_labels.append('reverb')
                        break
                    elif any(x in label for x in ['effect', 'fx', '效果']):
                        valid_labels.append('effect')
                        break
                    elif any(x in label for x in ['field', 'stage', 'space', '场', '声场', '空间']):
                        valid_labels.append('soundstage')
                        break
                    elif any(x in label for x in ['compress', '压', '压缩']):
                        valid_labels.append('compression')
                        break
                    elif any(x in label for x in ['volume', 'loud', '音量', '大小']):
                        valid_labels.append('volume')
                        break
            
            # 如果还是没有标签，使用整个文本进行推断
            if not valid_labels:
                text_lower = text.lower()
                if 'low' in text_lower or 'bass' in text_lower or '低' in text_lower:
                    valid_labels.append('low_freq')
                elif 'mid' in text_lower or 'middle' in text_lower or '中' in text_lower:
                    valid_labels.append('mid_freq')
                elif 'high' in text_lower or 'treble' in text_lower or '高' in text_lower:
                    valid_labels.append('high_freq')
                elif 'reverb' in text_lower or '混响' in text_lower:
                    valid_labels.append('reverb')
                elif 'effect' in text_lower or '效果' in text_lower:
                    valid_labels.append('effect')
                elif 'field' in text_lower or 'stage' in text_lower or '场' in text_lower:
                    valid_labels.append('soundstage')
                elif 'compress' in text_lower or '压' in text_lower:
                    valid_labels.append('compression')
                elif 'volume' in text_lower or '音量' in text_lower:
                    valid_labels.append('volume')
                else:
                    # 默认标签
                    valid_labels.append('high_freq')
            
            # 去重
            valid_labels = list(dict.fromkeys(valid_labels))
            print(f"最终标签: {valid_labels}")
            return valid_labels
        except Exception as e:
            print(f"❌ 解析标签失败: {str(e)}")
            return ['high_freq']  # 出错时使用默认标签

    def predict(self, input_text, lang="中文", temperature=0.75):  # 默认温度为0.75
        """预测混音标签"""
        try:
            # 系统消息
            system_message = "You are an audio processing expert who selects the most relevant parameter labels based on user descriptions."
            
            # 翻译输入文本
            # 注意：如果怀疑翻译导致问题，可以暂时禁用翻译
            # translated_text = input_text.lower()  # 直接使用原文
            translated_text = self._translate_text(input_text)
            
            print(f"输入文本: {input_text}")
            print(f"翻译后文本: {translated_text}")
            
            # 确保使用正确的选项名称，与训练时一致
            user_prompt = f"""Analyze audio processing needs and select relevant labels:

Please follow these standards:
1. Select only the most certain 1 label
2. Select additional labels only if very certain
3. Separate labels with commas
4. Must select at least one label

Options: low frequency/mid frequency/high frequency/reverb/effect/soundstage/compression/volume

Input: {translated_text}
Output:"""
            
            print(f"提示词: {user_prompt}")
            
            # 生成回答
            response = self._generate_response(user_prompt, system_message, temperature)
            
            # 解析标签
            parsed_labels = self._parse_labels(response)
            
            # 如果使用LLM预测失败，使用备用方法
            if not parsed_labels:
                print("⚠️ LLM解析失败，使用备用方法")
                # 备用方法：直接从输入文本中匹配关键词
                if '低频' in input_text or 'low' in input_text.lower() or 'bass' in input_text.lower():
                    parsed_labels = ['low_freq']
                elif '中频' in input_text or 'mid' in input_text.lower():
                    parsed_labels = ['mid_freq']
                elif '高频' in input_text or 'high' in input_text.lower() or 'treble' in input_text.lower():
                    parsed_labels = ['high_freq']
                elif '混响' in input_text or 'reverb' in input_text.lower() or 'echo' in input_text.lower():
                    parsed_labels = ['reverb']
                elif '效果' in input_text or 'effect' in input_text.lower():
                    parsed_labels = ['effect']
                elif '声场' in input_text or 'field' in input_text.lower() or 'stage' in input_text.lower():
                    parsed_labels = ['soundstage']
                elif '压缩' in input_text or 'compress' in input_text.lower():
                    parsed_labels = ['compression']
                elif '音量' in input_text or 'volume' in input_text.lower() or 'loud' in input_text.lower():
                    parsed_labels = ['volume']
                else:
                    parsed_labels = ['high_freq']  # 默认选择高频
            
            # 区分主标签和次要标签
            main_label = parsed_labels[0] if parsed_labels else ""
            secondary_labels = parsed_labels[1:] if len(parsed_labels) > 1 else []
            
            # 语言处理
            if lang == "中文":
                main_label_text = self.label_mapping[main_label]["zh"] if main_label else ""
                secondary_labels_text = "，".join([self.label_mapping[l]["zh"] for l in secondary_labels]) if secondary_labels else ""
            else:
                main_label_text = self.label_mapping[main_label]["en"] if main_label else ""
                secondary_labels_text = ", ".join([self.label_mapping[l]["en"] for l in secondary_labels]) if secondary_labels else ""
            
            result = (main_label_text, secondary_labels_text, ",".join(parsed_labels))
            print(f"预测结果: {result}")
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            return f"❌ 预测失败: {str(e)}", "", "ERROR"

# 测试案例
if __name__ == "__main__":
    try:
        predictor = MixingLabelPredictor()
        test_cases = [
            ("我想要声音更甜一点", ["soundstage", "effect"]),  # 测试与截图中的例子
            ("人声高频需要更明亮", ["high_freq"]),
            ("增加鼓组的空间感和压缩感", ["soundstage", "compression"]),
            ("整体低频太多", ["low_freq"]),
            ("The vocals need more air", ["high_freq"]),
            ("降低混响量", ["reverb"])
        ]

        for text, expected in test_cases:
            print("\n" + "="*50)
            print(f"测试: \"{text}\"")
            main_label, secondary_label, code = predictor.predict(text)
            print(f"主标签：{main_label}")
            if secondary_label:
                print(f"副标签：{secondary_label}")
            print(f"标签代码：{code}")
            print(f"预期代码：{','.join(expected)}")
            
            # 检查预测是否符合预期
            predicted_codes = code.split(",")
            if set(predicted_codes) == set(expected):
                print("✅ 测试通过")
            else:
                print("❌ 测试未通过")
            print("="*50)
                
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")