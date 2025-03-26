import json
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import threading

# ===== 配置和常量 =====
@dataclass
class ModelArguments:
    """模型参数配置"""
    model_name_or_path: str
    infer_backend: str = "huggingface"
    trust_remote_code: bool = True
    
@dataclass
class GeneratingArguments:
    """生成参数配置"""
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True

VALID_LABELS = {'low frequency', 'mid frequency', 'high frequency', 'reverb', 'effects', 'sound field', 'compression', 'volume'}
LABEL_MAPPING = {
    "低频": "low frequency",
    "中频": "mid frequency",
    "高频": "high frequency",
    "混响": "reverb",
    "效果器": "effects",
    "声场": "sound field",
    "压缩": "compression",
    "音量": "volume",
    "左边": "sound field",
    "右侧": "sound field"
}

# 不同温度配置
TEMPERATURE_CONFIGS = [
    {"name": "超低温度", "temperature": 0.01},
    {"name": "低温度", "temperature": 0.1},
    {"name": "中温度", "temperature": 0.3},
    {"name": "高温度", "temperature": 0.7},
    {"name": "超高温度", "temperature": 1.0}
]

# ===== 模板处理 =====
class Template:
    """提示词模板处理"""
    
    def __init__(self):
        # Deepseek模型模板格式
        self.template_type = "deepseek"
        self.system_prefix = "<|begin_of_system|>\n"
        self.system_suffix = "\n<|end_of_system|>"
        self.user_prefix = "\n<|begin_of_human|>\n"
        self.user_suffix = "\n<|end_of_human|>"
        self.assistant_prefix = "\n<|begin_of_assistant|>\n"
        self.assistant_suffix = "\n<|end_of_assistant|>"
        
        # ChatML模板格式
        # self.template_type = "chatml"
        # self.system_prefix = "<|im_start|>system\n"
        # self.system_suffix = "<|im_end|>"
        # self.user_prefix = "\n<|im_start|>user\n"
        # self.user_suffix = "<|im_end|>"
        # self.assistant_prefix = "\n<|im_start|>assistant\n"
        # self.assistant_suffix = "<|im_end|>"
    
    def encode_oneturn(self, tokenizer, messages, system=None):
        """处理单轮对话并编码为token序列"""
        # 构建系统提示
        prompt = ""
        if system:
            prompt += self.system_prefix + system + self.system_suffix
        
        # 处理用户消息
        user_content = messages[0]["content"] if len(messages) > 0 else ""
        prompt += self.user_prefix + user_content + self.user_suffix
        
        # 添加助手前缀
        prompt += self.assistant_prefix
        
        # 编码为token
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        return inputs, len(inputs.input_ids[0])
    
    def get_stop_token_ids(self, tokenizer):
        """获取终止token ID"""
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

# ===== 引擎实现 =====
class BaseEngine:
    """基础引擎接口"""
    
    def __init__(self, model_args, generating_args, device="cuda"):
        self.model_args = model_args
        self.generating_args = generating_args
        self.device = device
        self.template = Template()
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        raise NotImplementedError
    
    def chat(self, messages, system=None):
        """进行对话"""
        raise NotImplementedError
    
    def parse_labels(self, text):
        """解析标签"""
        if not text or not isinstance(text, str):
            return []
            
        text = text.strip().lower().replace(" ", "").replace("-", "")
        
        # 处理英文和中文标点
        raw_labels = [label.strip() for label in re.split(r'[,，、;/]', text) if label.strip()]
        
        validated_labels = []
        for label in raw_labels:
            # 直接匹配
            if label in VALID_LABELS:
                validated_labels.append(label)
                continue
            if label in LABEL_MAPPING:
                validated_labels.append(LABEL_MAPPING[label])
                continue
            
            # 模糊匹配
            if any(x in label for x in ['sound', 'field', '声场']):
                validated_labels.append('sound field')
            elif any(x in label for x in ['compress', '压', '缩']):
                validated_labels.append('compression')
            elif any(x in label for x in ['effect', '效果']):
                validated_labels.append('effects')
            elif any(x in label for x in ['reverb', '混响']):
                validated_labels.append('reverb')
            elif any(x in label for x in ['low', '低']) and any(x in label for x in ['freq', '频']):
                validated_labels.append('low frequency')
            elif any(x in label for x in ['mid', '中']) and any(x in label for x in ['freq', '频']):
                validated_labels.append('mid frequency')
            elif any(x in label for x in ['high', '高']) and any(x in label for x in ['freq', '频']):
                validated_labels.append('high frequency')
            elif any(x in label for x in ['volume', '音量']):
                validated_labels.append('volume')
        
        valid_labels = list(set(validated_labels) & VALID_LABELS)
        
        # 确保至少有一个标签
        if not valid_labels:
            # 尝试从文本中推断一个最可能的标签
            if '低' in text or 'low' in text or 'bass' in text:
                valid_labels = ['low frequency']
            elif '中' in text or 'mid' in text:
                valid_labels = ['mid frequency']
            elif '高' in text or 'high' in text or 'treble' in text:
                valid_labels = ['high frequency']
            elif '混响' in text or 'reverb' in text or 'echo' in text:
                valid_labels = ['reverb']
            elif '效果' in text or 'effect' in text:
                valid_labels = ['effects']
            elif '场' in text or 'field' in text or 'space' in text or 'wide' in text or '宽' in text:
                valid_labels = ['sound field']
            elif '压' in text or 'compress' in text:
                valid_labels = ['compression']
            elif '音量' in text or 'volume' in text or 'loud' in text:
                valid_labels = ['volume']
            else:
                # 如果还是无法推断，随机选择一个标签
                valid_labels = [random.choice(list(VALID_LABELS))]
                print(f"警告: 无法从'{text}'中提取有效标签，随机选择'{valid_labels[0]}'")
        
        return valid_labels

class HuggingfaceEngine(BaseEngine):
    """Huggingface引擎实现"""
    
    def load_model(self):
        """加载模型与分词器"""
        print(f"正在加载模型: {self.model_args.model_name_or_path}")
        
        try:
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=self.model_args.trust_remote_code
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path,
                trust_remote_code=self.model_args.trust_remote_code,
                padding_side="left",
                truncation_side="left"
            )
            
            # 确保有填充token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def chat(self, messages, system=None, temperature=None):
        """聊天功能实现，可选温度参数"""
        # 预处理消息
        inputs, prompt_length = self.template.encode_oneturn(
            self.tokenizer, messages, system
        )
        inputs = inputs.to(self.device)
        
        # 使用提供的温度或默认值
        temp = temperature if temperature is not None else self.generating_args.temperature
        
        # 创建生成配置
        generation_config = GenerationConfig(
            temperature=temp,
            top_p=self.generating_args.top_p,
            max_new_tokens=self.generating_args.max_new_tokens,
            repetition_penalty=self.generating_args.repetition_penalty,
            do_sample=self.generating_args.do_sample,
            eos_token_id=self.template.get_stop_token_ids(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 生成回复
        with torch.no_grad():
            generate_output = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )
        
        # 解码响应
        response_ids = generate_output[0, prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip()

# ===== 聊天模型 =====
class ChatModel:
    """聊天模型封装"""
    
    def __init__(self, model_path):
        # 初始化参数
        model_args = ModelArguments(
            model_name_or_path=model_path,
            infer_backend="huggingface",
            trust_remote_code=True
        )
        
        generating_args = GeneratingArguments(
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=50,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        # 创建引擎
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = HuggingfaceEngine(model_args, generating_args, self.device)
        self.initialized = self.engine.load_model()
    
    def chat(self, user_input, system_message=None, temperature=None):
        """进行聊天"""
        if not self.initialized:
            return "模型未成功初始化"
        
        # 创建消息
        message = [{"role": "user", "content": user_input}]
        
        # 调用引擎聊天
        response = self.engine.chat(message, system=system_message, temperature=temperature)
        
        return response
    
    def analyze_audio(self, description, temperature=None):
        """分析音频处理需求"""
        system_message = "你是一个音频处理专家，需要根据用户描述选择最相关的参数标签。"
        
        user_input = f"""分析音频处理需求并选择相关标签：

请按照以下标准进行选择：
1. 仅选择最确定的1个标签
2. 只有在非常确定的情况下才选择额外标签
3. 标签之间用逗号分隔
4. 必须至少选择一个标签

选项：低频/中频/高频/reverb/效果器/声场/压缩/音量

Input: {description}
Output:"""
        
        # 记录完整提示词
        print(f"提示词: {user_input}")
        
        response = self.chat(user_input, system_message, temperature)
        
        # 记录原始响应
        print(f"原始响应: '{response}'")
        
        # 解析标签
        parsed_labels = self.engine.parse_labels(response)
        
        # 确保至少有一个标签
        if not parsed_labels:
            # 根据输入描述推断最可能的标签
            if '低' in description or 'low' in description or 'bass' in description:
                parsed_labels = ['low frequency']
            elif '中' in description or 'mid' in description:
                parsed_labels = ['mid frequency']
            elif '高' in description or 'high' in description or 'treble' in description:
                parsed_labels = ['high frequency']
            elif '混响' in description or 'reverb' in description or 'echo' in description:
                parsed_labels = ['reverb']
            elif '效果' in description or 'effect' in description:
                parsed_labels = ['effects']
            elif '场' in description or 'field' in description or 'space' in description or 'wide' in description:
                parsed_labels = ['sound field']
            elif '压' in description or 'compress' in description:
                parsed_labels = ['compression']
            elif '音量' in description or 'volume' in description or 'loud' in description:
                parsed_labels = ['volume']
            else:
                # 随机选择一个标签
                parsed_labels = [random.choice(list(VALID_LABELS))]
            
            print(f"发现空标签，已填充为: {parsed_labels}")
        
        return parsed_labels, response

# ===== 评估函数 =====
def calculate_metrics(true_labels, predicted_labels):
    """计算评估指标，返回字典"""
    # 精确匹配
    exact_match = set(true_labels) == set(predicted_labels)
    
    # 部分匹配 - 至少有一个标签匹配
    partial_match = len(set(true_labels) & set(predicted_labels)) > 0
    
    # 召回率 - 真实标签中有多少被正确预测
    recall = 0
    if len(true_labels) > 0:
        recall = len(set(true_labels) & set(predicted_labels)) / len(true_labels)
    
    # 精确率 - 预测标签中有多少是正确的
    precision = 0
    if len(predicted_labels) > 0:
        precision = len(set(true_labels) & set(predicted_labels)) / len(predicted_labels)
    
    # F1分数
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "exact_match": exact_match,
        "partial_match": partial_match,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "true_count": len(true_labels),
        "pred_count": len(predicted_labels),
        "correct_count": len(set(true_labels) & set(predicted_labels))
    }

# ===== 测试函数 =====
def test_audio_analysis(model_path, test_data, output_dir):
    """测试音频分析功能"""
    print("开始音频分析测试")
    
    # 创建聊天模型
    chat_model = ChatModel(model_path)
    if not chat_model.initialized:
        print("模型初始化失败，测试终止")
        return
    
    # 为每个温度配置创建结果集
    all_temp_results = {}
    
    for temp_config in TEMPERATURE_CONFIGS:
        temp_name = temp_config["name"]
        temperature = temp_config["temperature"]
        print(f"\n\n== 测试温度配置: {temp_name} (temperature={temperature}) ==")
        
        # 测试结果
        results = []
        correct_count = 0
        partial_match_count = 0
        single_label_count = 0
        single_label_correct = 0
        total_count = len(test_data)
        
        # 按标签统计
        label_stats = {label: {"correct": 0, "total": 0, "partial": 0} for label in VALID_LABELS}
        
        # 测试每个样本
        for i, sample in enumerate(test_data):
            input_text = sample.get('input', '')
            true_labels = chat_model.engine.parse_labels(sample.get('output', ''))
            
            print(f"\n测试样本 {i+1}/{total_count}: '{input_text}'")
            print(f"标准答案: {true_labels}")
            
            # 统计各标签出现次数
            for label in true_labels:
                label_stats[label]["total"] += 1
            
            # 是否单标签样本
            is_single_label = len(true_labels) == 1
            if is_single_label:
                single_label_count += 1
            
            try:
                # 分析音频
                predicted_labels, raw_response = chat_model.analyze_audio(input_text, temperature)
                predicted_copy = predicted_labels.copy()  # 立即创建副本
                
                # 确保预测结果非空
                assert len(predicted_copy) > 0, "预测结果不能为空"
                
                # 计算评估指标
                metrics = calculate_metrics(true_labels, predicted_copy)
                
                # 统计匹配情况
                if metrics["exact_match"]:
                    correct_count += 1
                    print(f"✅ 预测正确: {predicted_copy}")
                    
                    # 更新标签统计
                    for label in true_labels:
                        label_stats[label]["correct"] += 1
                    
                    # 单标签正确统计
                    if is_single_label:
                        single_label_correct += 1
                        
                elif metrics["partial_match"]:
                    partial_match_count += 1
                    print(f"⚠️ 部分正确: {predicted_copy} (命中 {metrics['correct_count']}/{metrics['true_count']})")
                    
                    # 更新部分匹配标签统计
                    for label in set(true_labels) & set(predicted_copy):
                        label_stats[label]["partial"] += 1
                    
                else:
                    print(f"❌ 预测错误: {predicted_copy}")
                
                # 记录结果
                result_entry = {
                    "input": input_text,
                    "true_labels": true_labels.copy(),
                    "predicted_labels": predicted_copy,
                    "raw_response": raw_response,
                    "metrics": metrics,
                    "is_single_label": is_single_label
                }
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"分析失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 计算准确率
        exact_match_rate = correct_count / total_count if total_count > 0 else 0
        partial_match_rate = partial_match_count / total_count if total_count > 0 else 0
        single_label_accuracy = single_label_correct / single_label_count if single_label_count > 0 else 0
        
        # 计算标签准确率
        label_accuracy = {}
        for label, stats in label_stats.items():
            if stats["total"] > 0:
                exact_rate = stats["correct"] / stats["total"]
                partial_rate = (stats["correct"] + stats["partial"]) / stats["total"]
                label_accuracy[label] = {
                    "exact_match": f"{stats['correct']}/{stats['total']} ({exact_rate*100:.1f}%)",
                    "partial_match": f"{stats['correct'] + stats['partial']}/{stats['total']} ({partial_rate*100:.1f}%)",
                    "correct": stats["correct"],
                    "partial": stats["partial"],
                    "total": stats["total"]
                }
            else:
                label_accuracy[label] = {
                    "exact_match": "0/0 (0.0%)",
                    "partial_match": "0/0 (0.0%)",
                    "correct": 0,
                    "partial": 0,
                    "total": 0
                }
        
        # 打印总结果
        print(f"\n{temp_name} (temperature={temperature}) 测试完成! 总结:")
        print(f"总样本数: {total_count}")
        print(f"精确匹配: {correct_count}/{total_count} ({exact_match_rate*100:.1f}%)")
        print(f"部分匹配: {partial_match_count}/{total_count} ({partial_match_rate*100:.1f}%)")
        print(f"总匹配率: {(correct_count + partial_match_count)}/{total_count} ({(exact_match_rate + partial_match_rate)*100:.1f}%)")
        print(f"单标签样本准确率: {single_label_correct}/{single_label_count} ({single_label_accuracy*100:.1f}%)")
        
        # 打印各标签准确率
        print("\n各标签精确匹配率:")
        for label, acc in label_accuracy.items():
            if acc["total"] > 0:
                print(f"{label}: {acc['exact_match']}")
        
        print("\n各标签部分匹配率:")
        for label, acc in label_accuracy.items():
            if acc["total"] > 0:
                print(f"{label}: {acc['partial_match']}")
        
        # 保存温度配置结果
        all_temp_results[temp_name] = {
            "temperature": temperature,
            "total": total_count,
            "exact_match": {
                "count": correct_count,
                "rate": exact_match_rate
            },
            "partial_match": {
                "count": partial_match_count,
                "rate": partial_match_rate
            },
            "total_match": {
                "count": correct_count + partial_match_count,
                "rate": exact_match_rate + partial_match_rate
            },
            "single_label": {
                "count": single_label_count,
                "correct": single_label_correct,
                "rate": single_label_accuracy
            },
            "label_accuracy": label_accuracy,
            "results": results
        }
    
    # 找出最佳温度配置
    best_exact_temp = None
    best_exact_rate = -1
    best_partial_temp = None
    best_partial_rate = -1
    
    for temp_name, results in all_temp_results.items():
        exact_rate = results["exact_match"]["rate"]
        partial_rate = results["total_match"]["rate"]
        
        if exact_rate > best_exact_rate:
            best_exact_rate = exact_rate
            best_exact_temp = temp_name
            
        if partial_rate > best_partial_rate:
            best_partial_rate = partial_rate
            best_partial_temp = temp_name
    
    print("\n\n最佳温度配置:")
    print(f"精确匹配最佳: {best_exact_temp} (temperature={all_temp_results[best_exact_temp]['temperature']}) - 准确率: {best_exact_rate*100:.1f}%")
    print(f"部分匹配最佳: {best_partial_temp} (temperature={all_temp_results[best_partial_temp]['temperature']}) - 准确率: {best_partial_rate*100:.1f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"temperature_comparison_results_{timestamp}.json")
    
    # 构建结果对象
    output_data = {
        "model": model_path,
        "timestamp": timestamp,
        "best_temp": {
            "exact_match": {
                "name": best_exact_temp,
                "temperature": all_temp_results[best_exact_temp]["temperature"],
                "rate": best_exact_rate
            },
            "partial_match": {
                "name": best_partial_temp,
                "temperature": all_temp_results[best_partial_temp]["temperature"],
                "rate": best_partial_rate
            }
        },
        "temperature_results": all_temp_results
    }
    
    # 保存为JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存至: {output_file}")
    return output_data["best_temp"]
    
# ===== 主函数 =====
def main():
    # 模型路径
    model_path = r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v5"
    
    # 输出目录
    output_dir = r"D:\kings\prj\MixMaster-Finetune\eval"
    
    # 加载测试数据
    test_data_path = r"D:\kings\prj\MixMaster-Finetune\data\eval\eval_data.json"
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"成功加载 {len(test_data)} 条测试数据")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        test_data = []
        return
    
    # 测试音频分析
    test_audio_analysis(model_path, test_data, output_dir)

if __name__ == "__main__":
    main()