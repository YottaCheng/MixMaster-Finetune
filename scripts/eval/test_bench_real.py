import streamlit as st
import json
import torch
import os
import re
import time
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm.auto import tqdm
import threading
from dataclasses import dataclass

# Constants
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
    "左边": "sound field",  # Additional mapping from first code
    "右侧": "sound field"   # Additional mapping from first code
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Model template for formatting prompts (borrowed from first code)
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

# Generation arguments configuration (from first code)
@dataclass
class GeneratingArguments:
    """Generation parameter configuration"""
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True

# Temperature configurations for testing (from first code)
TEMPERATURE_CONFIGS = [
    {"name": "Ultra-low temperature", "temperature": 0.01},
    {"name": "Low temperature", "temperature": 0.1},
    {"name": "Medium temperature", "temperature": 0.3},
    {"name": "High temperature", "temperature": 0.7},
    {"name": "Ultra-high temperature", "temperature": 1.0}
]

# Utility functions
def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_labels(text):
    """Parse and standardize labels (enhanced version combining both implementations)"""
    if not text or not isinstance(text, str):
        return []
    
    # Clean and standardize text
    text = text.strip().lower().replace(" ", "").replace("-", "")
    
    # Handle both English and Chinese punctuation
    raw_labels = [label.strip() for label in re.split(r'[,，、;/]', text) if label.strip()]
    
    # Validate labels
    validated_labels = []
    for label in raw_labels:
        # Direct matching
        if label in VALID_LABELS:
            validated_labels.append(label)
            continue
        if label in LABEL_MAPPING:
            validated_labels.append(LABEL_MAPPING[label])
            continue
        
        # Fuzzy matching (from first code)
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
    
    # Filter to valid labels
    valid_labels = list(set(validated_labels) & VALID_LABELS)
    
    # Ensure at least one label (fallback logic from first code)
    if not valid_labels:
        # Try to infer the most likely label from text
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
    
    return valid_labels

def extract_prediction(text, prompt_template, input_text):
    """Extract prediction labels from model-generated text"""
    # Create the full prompt that was used
    prompt = prompt_template.replace("[TEXT]", input_text)
    
    # Remove prompt part if present
    if prompt in text:
        text = text[len(prompt):]
    
    # Try to find the "Output:" part
    if "Output:" in text:
        text = text.split("Output:")[1].strip()
    elif "output:" in text:
        text = text.split("output:")[1].strip()
    elif "输出：" in text:
        text = text.split("输出：")[1].strip()
    
    # Clean the text
    text = re.sub(r'^[\s:"：]+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    
    # Parse labels
    return parse_labels(text)

def calculate_metrics(true_labels, predicted_labels):
    """Calculate evaluation metrics, returns dictionary (from first code)"""
    # Exact match
    exact_match = set(true_labels) == set(predicted_labels)
    
    # Partial match - at least one label matches
    partial_match = len(set(true_labels) & set(predicted_labels)) > 0
    
    # Recall - percentage of true labels correctly predicted
    recall = 0
    if len(true_labels) > 0:
        recall = len(set(true_labels) & set(predicted_labels)) / len(true_labels)
    
    # Precision - percentage of predicted labels that are correct
    precision = 0
    if len(predicted_labels) > 0:
        precision = len(set(true_labels) & set(predicted_labels)) / len(predicted_labels)
    
    # F1 score
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

def create_evaluation_metrics(all_true_labels, all_pred_labels, label_set):
    """Calculate multi-label evaluation metrics (from second code)"""
    # Build binary arrays
    y_true = []
    y_pred = []
    
    for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
        true_vec = [1 if label in true_labels else 0 for label in label_set]
        pred_vec = [1 if label in pred_labels else 0 for label in label_set]
        y_true.append(true_vec)
        y_pred.append(pred_vec)
    
    # Calculate label-level metrics
    label_metrics = {}
    for i, label in enumerate(label_set):
        true_label = [row[i] for row in y_true]
        pred_label = [row[i] for row in y_pred]
        
        # Avoid division by zero
        if sum(true_label) == 0 and sum(pred_label) == 0:
            precision = 1.0
        elif sum(true_label) == 0:
            precision = 0.0
        elif sum(pred_label) == 0:
            precision = 0.0
        else:
            precision = precision_score(true_label, pred_label, zero_division=0)
            
        recall = recall_score(true_label, pred_label, zero_division=0)
        f1 = f1_score(true_label, pred_label, zero_division=0)
        
        label_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(true_label)
        }
    
    # Sample-level metrics
    sample_accuracy = sum(1 for t, p in zip(all_true_labels, all_pred_labels) 
                          if len(set(t) & set(p)) > 0) / max(1, len(all_true_labels))
    
    exact_match = sum(1 for t, p in zip(all_true_labels, all_pred_labels) 
                     if set(t) == set(p)) / max(1, len(all_true_labels))
    
    # Label-level metrics
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'sample_metrics': {
            'accuracy': sample_accuracy,
            'exact_match': exact_match
        },
        'micro_metrics': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'macro_metrics': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'label_metrics': label_metrics,
        'sample_count': len(all_true_labels)
    }

# Enhanced model loading and evaluation
class EnhancedModelEvaluator:
    """Enhanced model evaluator that incorporates features from both codebases"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.template = Template()  # Using first code's template
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            # Load model (from first code)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left"
            )
            
            # Ensure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def predict(self, input_text, system_message=None, temperature=0.1):
        """Generate prediction for input text with specified temperature"""
        if not self.initialized:
            return "Model not initialized"
        
        # Create message
        message = [{"role": "user", "content": input_text}]
        
        # Encode message
        inputs, prompt_length = self.template.encode_oneturn(
            self.tokenizer, message, system_message
        )
        inputs = inputs.to(self.device)
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=50,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=self.template.get_stop_token_ids(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Generate response
        with torch.no_grad():
            generate_output = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )
        
        # Decode response
        response_ids = generate_output[0, prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def analyze_audio(self, description, system_message=None, temperature=None):
        """Analyze audio processing needs"""
        if not system_message:
            system_message = "You are an audio processing expert who selects the most relevant parameter labels based on user descriptions."
        
        user_input = f"""Analyze audio processing needs and select relevant labels:

Please follow these standards:
1. Select only the most certain 1 label
2. Select additional labels only if very certain
3. Separate labels with commas
4. Must select at least one label

Options: low frequency/mid frequency/high frequency/reverb/effects/sound field/compression/volume

Input: {description}
Output:"""
        
        # Generate response
        response = self.predict(user_input, system_message, temperature)
        
        # Parse labels
        parsed_labels = parse_labels(response)
        
        return parsed_labels, response

def run_optimized_evaluation(test_data, model_paths, max_samples=50, seed=42):
    """Run optimized model evaluation with multiple temperature settings"""
    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_estimate = st.empty()
    
    # Set random seed
    set_seed(seed)
    
    # Process test data
    status_text.text("Preparing test data...")
    if max_samples and max_samples < len(test_data):
        test_samples = random.sample(test_data, max_samples)
    else:
        test_samples = test_data
    progress_bar.progress(5)
    
    # Extract true labels
    true_labels_by_sample = []
    for sample in test_samples:
        output_text = sample.get('output', '')
        true_labels = parse_labels(output_text)
        true_labels_by_sample.append(true_labels)
    
    # Load base model
    status_text.text("Loading base model...")
    base_evaluator = EnhancedModelEvaluator(model_paths["base"])
    if not base_evaluator.load_model():
        status_text.error("Failed to load base model")
        return None, None
    progress_bar.progress(15)
    
    # Load fine-tuned model
    status_text.text("Loading fine-tuned model...")
    ft_evaluator = EnhancedModelEvaluator(model_paths["finetuned"])
    if not ft_evaluator.load_model():
        status_text.error("Failed to load fine-tuned model")
        return None, None
    progress_bar.progress(25)
    
    # Create result containers
    examples = []
    all_results = {
        "base_model": {
            "defaults": None,
            "by_temperature": {}
        },
        "finetuned_model": {
            "defaults": None,
            "by_temperature": {}
        }
    }
    
    # Track timings for estimation
    start_time = time.time()
    sample_times = []
    
    # Set total progress segments
    total_progress_segments = 75  # We've used 25 already
    progress_per_step = total_progress_segments / (len(test_samples) * (1 + len(TEMPERATURE_CONFIGS)))
    current_progress = 25
    
    # First evaluate with default temperatures
    status_text.text("Running default evaluation...")
    
    base_preds = []
    ft_preds = []
    
    # Default temperatures
    base_default_temp = 0.7  # Higher for base model
    ft_default_temp = 0.1    # Lower for fine-tuned model
    
    # Process all samples
    for i, sample in enumerate(test_samples):
        sample_start_time = time.time()
        
        # Update progress
        status_text.text(f"Default evaluation: sample {i+1}/{len(test_samples)}...")
        
        try:
            input_text = sample['input']
            true_labels = true_labels_by_sample[i]
            
            # Get predictions with default temperatures
            base_prediction, base_response = base_evaluator.analyze_audio(input_text, temperature=base_default_temp)
            ft_prediction, ft_response = ft_evaluator.analyze_audio(input_text, temperature=ft_default_temp)
            
            # Store predictions
            base_preds.append(base_prediction)
            ft_preds.append(ft_prediction)
            
            # Store example
            examples.append({
                'input': input_text,
                'true_labels': true_labels,
                'base_prediction': base_prediction,
                'ft_prediction': ft_prediction,
                'base_full_response': base_response,
                'ft_full_response': ft_response
            })
            
            # Update progress
            current_progress += progress_per_step
            progress_bar.progress(int(current_progress))
            
            # Update time estimation
            sample_time = time.time() - sample_start_time
            sample_times.append(sample_time)
            avg_time = sum(sample_times) / len(sample_times)
            remaining_samples = len(test_samples) - (i + 1)
            remaining_time = avg_time * remaining_samples
            
            # Format time display
            if remaining_time > 60:
                time_estimate.text(f"Estimated time for default evaluation: {int(remaining_time/60)} min {int(remaining_time%60)} sec")
            else:
                time_estimate.text(f"Estimated time for default evaluation: {int(remaining_time)} sec")
                
        except Exception as e:
            st.warning(f"Error processing sample {i+1}: {str(e)}")
            continue
    
    # Calculate default metrics
    base_metrics = create_evaluation_metrics(true_labels_by_sample, base_preds, list(VALID_LABELS))
    ft_metrics = create_evaluation_metrics(true_labels_by_sample, ft_preds, list(VALID_LABELS))
    
    all_results["base_model"]["defaults"] = base_metrics
    all_results["finetuned_model"]["defaults"] = ft_metrics
    
    # Now test different temperatures (optional - could be toggled by user)
    with st.expander("Advanced Temperature Analysis", expanded=False):
        run_temperature_analysis = st.checkbox("Run temperature analysis", value=False)
    
    if not run_temperature_analysis:
        # Skip temperature analysis
        progress_bar.progress(100)
        status_text.text("Evaluation complete!")
        time_estimate.text(f"Total evaluation time: {int(time.time() - start_time)} seconds")
        return all_results, examples
    
    # Run temperature analysis for fine-tuned model only
    status_text.text("Running temperature analysis...")
    
    # Reset progress tracking for temperature analysis
    temp_progress_segments = total_progress_segments / len(TEMPERATURE_CONFIGS)
    
    # Test each temperature configuration
    for temp_idx, temp_config in enumerate(TEMPERATURE_CONFIGS):
        temp_name = temp_config["name"]
        temperature = temp_config["temperature"]
        
        status_text.text(f"Testing temperature: {temp_name} ({temperature})...")
        
        # Reset predictions for this temperature
        ft_temp_preds = []
        
        # Process samples with this temperature
        for i, sample in enumerate(test_samples):
            try:
                input_text = sample['input']
                
                # Get predictions with this temperature
                ft_prediction, _ = ft_evaluator.analyze_audio(input_text, temperature=temperature)
                ft_temp_preds.append(ft_prediction)
                
            except Exception as e:
                st.warning(f"Error processing sample with temp {temp_name}: {str(e)}")
                continue
        
        # Calculate metrics for this temperature
        ft_temp_metrics = create_evaluation_metrics(true_labels_by_sample, ft_temp_preds, list(VALID_LABELS))
        all_results["finetuned_model"]["by_temperature"][temp_name] = {
            "temperature": temperature,
            "metrics": ft_temp_metrics
        }
        
        # Update progress
        current_progress += temp_progress_segments
        progress_bar.progress(int(min(100, current_progress)))
    
    # Find best temperature configuration
    best_exact_temp = None
    best_exact_rate = -1
    
    for temp_name, temp_data in all_results["finetuned_model"]["by_temperature"].items():
        exact_rate = temp_data["metrics"]["sample_metrics"]["exact_match"]
        
        if exact_rate > best_exact_rate:
            best_exact_rate = exact_rate
            best_exact_temp = temp_name
    
    # Add best temperature info
    all_results["finetuned_model"]["best_temperature"] = {
        "name": best_exact_temp,
        "temperature": all_results["finetuned_model"]["by_temperature"][best_exact_temp]["temperature"],
        "exact_match": best_exact_rate
    }
    
    # Complete evaluation
    progress_bar.progress(100)
    status_text.text("Evaluation complete!")
    time_estimate.text(f"Total evaluation time: {int(time.time() - start_time)} seconds")
    
    return all_results, examples

def display_optimized_results(results, examples, true_labels_by_sample=None, ft_preds=None):
    """Display enhanced evaluation results"""
    # 如果参数为None，从examples中提取数据
    if true_labels_by_sample is None:
        true_labels_by_sample = [ex['true_labels'] for ex in examples]
    
    if ft_preds is None:
        ft_preds = [ex['ft_prediction'] for ex in examples]
    
    # 显示默认比较
    base_metrics = results["base_model"]["defaults"]
    ft_metrics = results["finetuned_model"]["defaults"]
    
    # 主要指标
    st.markdown("## Main Evaluation Results")
    
    # 指标卡片行
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Exact Match Rate</div>', unsafe_allow_html=True)
        
        base_exact = base_metrics['sample_metrics']['exact_match']
        ft_exact = ft_metrics['sample_metrics']['exact_match']
        change = ft_exact - base_exact
        change_pct = (change / base_exact * 100) if base_exact > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_exact:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">vs. Base Model: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Partial Match Rate</div>', unsafe_allow_html=True)
        
        base_acc = base_metrics['sample_metrics']['accuracy']
        ft_acc = ft_metrics['sample_metrics']['accuracy']
        change = ft_acc - base_acc
        change_pct = (change / base_acc * 100) if base_acc > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_acc:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">vs. Base Model: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Micro F1 Score</div>', unsafe_allow_html=True)
        
        base_f1 = base_metrics['micro_metrics']['f1']
        ft_f1 = ft_metrics['micro_metrics']['f1']
        change = ft_f1 - base_f1
        change_pct = (change / base_f1 * 100) if base_f1 > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_f1:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">vs. Base Model: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Temperature analysis results
    if "best_temperature" in results["finetuned_model"]:
        st.markdown("## Temperature Analysis")
        
        best_temp = results["finetuned_model"]["best_temperature"]
        
        st.success(f"Best Temperature Configuration: **{best_temp['name']}** (temperature={best_temp['temperature']}) with Exact Match Rate: {best_temp['exact_match']:.2%}")
        
        # Create temperature comparison chart
        temp_data = []
        for temp_name, temp_info in results["finetuned_model"]["by_temperature"].items():
            temp_data.append({
                "Temperature": temp_name,
                "Exact Match": temp_info["metrics"]["sample_metrics"]["exact_match"],
                "Partial Match": temp_info["metrics"]["sample_metrics"]["accuracy"],
                "Micro F1": temp_info["metrics"]["micro_metrics"]["f1"],
                "Value": temp_info["temperature"]
            })
        
        temp_df = pd.DataFrame(temp_data)
        
        # Create temperature performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[d["Temperature"] for d in temp_data],
            y=[d["Exact Match"] for d in temp_data],
            name='Exact Match',
            marker_color='#4CAF50'
        ))
        
        fig.add_trace(go.Bar(
            x=[d["Temperature"] for d in temp_data],
            y=[d["Partial Match"] for d in temp_data],
            name='Partial Match',
            marker_color='#2196F3'
        ))
        
        fig.add_trace(go.Bar(
            x=[d["Temperature"] for d in temp_data],
            y=[d["Micro F1"] for d in temp_data],
            name='Micro F1',
            marker_color='#9C27B0'
        ))
        
        fig.update_layout(
            title='Performance Across Temperature Settings',
            xaxis=dict(title='Temperature'),
            yaxis=dict(title='Score', tickformat='.0%'),
            barmode='group',
            legend=dict(x=0.1, y=1.1, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed temperature metrics
        st.markdown("### Temperature Metrics Comparison")
        
        # Format the DataFrame for display
        display_df = temp_df.copy()
        for col in ['Exact Match', 'Partial Match', 'Micro F1']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        
        # Add the actual temperature value
        display_df = display_df.sort_values('Value')
        
        st.dataframe(display_df)
    
    # Label-specific performance
    st.markdown("## Label-specific Performance")
    
    # Create label metrics comparison
    label_data = []
    
    for label in VALID_LABELS:
        base_label_metrics = base_metrics['label_metrics'][label]
        ft_label_metrics = ft_metrics['label_metrics'][label]
        
        label_data.append({
            "Label": label,
            "Base F1": base_label_metrics['f1'],
            "Fine-tuned F1": ft_label_metrics['f1'],
            "Improvement": ft_label_metrics['f1'] - base_label_metrics['f1'],
            "Improvement %": ((ft_label_metrics['f1'] - base_label_metrics['f1']) / max(0.001, base_label_metrics['f1'])) * 100,
            "Support": base_label_metrics['support']
        })
    
    label_df = pd.DataFrame(label_data)
    
    # Sort by improvement
    label_df = label_df.sort_values("Improvement", ascending=False)
    
    # Create label performance chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=label_df["Label"],
        y=label_df["Base F1"],
        name='Base Model',
        marker_color='#90CAF9'
    ))
    
    fig.add_trace(go.Bar(
        x=label_df["Label"],
        y=label_df["Fine-tuned F1"],
        name='Fine-tuned Model',
        marker_color='#F48FB1'
    ))
    
    fig.update_layout(
        title='F1 Score by Label',
        xaxis=dict(title='Label'),
        yaxis=dict(title='F1 Score', tickformat='.0%'),
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display label metrics table
    st.markdown("### Label-specific Metrics Table")
    
    # Format the DataFrame for display
    display_label_df = label_df.copy()
    for col in ['Base F1', 'Fine-tuned F1']:
        display_label_df[col] = display_label_df[col].apply(lambda x: f"{x:.2%}")
    
    display_label_df["Improvement %"] = display_label_df["Improvement %"].apply(lambda x: f"{x:+.2f}%" if not np.isinf(x) else "N/A")
    
    st.dataframe(display_label_df)
    
    # Prediction examples
    st.markdown("## Prediction Examples")
    
    # Success examples tab
    tab1, tab2, tab3 = st.tabs(["Success Cases", "Improvement Cases", "Failure Cases"])
    
    with tab1:
        # Filter for exact matches by the fine-tuned model
        success_examples = [ex for ex in examples if set(ex['true_labels']) == set(ex['ft_prediction'])]
        
        if success_examples:
            for i, example in enumerate(success_examples[:5]):
                with st.expander(f"Example {i+1}: {example['input'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**Input Text**: {example['input']}")
                    st.markdown(f"**True Labels**: {', '.join(example['true_labels'])}")
                    st.markdown(f"**Fine-tuned Model Prediction**: ✅ {', '.join(example['ft_prediction'])}")
        else:
            st.info("No success cases found in the evaluated samples.")
    
    with tab2:
        # Filter for cases where fine-tuned model was correct but base model was wrong
        improvement_examples = [
            ex for ex in examples 
            if set(ex['true_labels']) == set(ex['ft_prediction']) and set(ex['true_labels']) != set(ex['base_prediction'])
        ]
        
        if improvement_examples:
            for i, example in enumerate(improvement_examples[:5]):
                with st.expander(f"Example {i+1}: {example['input'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**Input Text**: {example['input']}")
                    st.markdown(f"**True Labels**: {', '.join(example['true_labels'])}")
                    st.markdown(f"**Base Model Prediction**: ❌ {', '.join(example['base_prediction'])}")
                    st.markdown(f"**Fine-tuned Model Prediction**: ✅ {', '.join(example['ft_prediction'])}")
        else:
            st.info("No improvement cases found in the evaluated samples.")
    
    with tab3:
        # Filter for cases where fine-tuned model was wrong
        failure_examples = [ex for ex in examples if set(ex['true_labels']) != set(ex['ft_prediction'])]
        
        if failure_examples:
            for i, example in enumerate(failure_examples[:5]):
                with st.expander(f"Example {i+1}: {example['input'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**Input Text**: {example['input']}")
                    st.markdown(f"**True Labels**: {', '.join(example['true_labels'])}")
                    st.markdown(f"**Fine-tuned Model Prediction**: ❌ {', '.join(example['ft_prediction'])}")
                    
                    # Calculate partially correct labels
                    correct_labels = set(example['true_labels']) & set(example['ft_prediction'])
                    missed_labels = set(example['true_labels']) - set(example['ft_prediction'])
                    extra_labels = set(example['ft_prediction']) - set(example['true_labels'])
                    
                    if correct_labels:
                        st.markdown(f"**Correct Labels**: ✅ {', '.join(correct_labels)}")
                    if missed_labels:
                        st.markdown(f"**Missed Labels**: ❓ {', '.join(missed_labels)}")
                    if extra_labels:
                        st.markdown(f"**Extra Labels**: ❌ {', '.join(extra_labels)}")
        else:
            st.info("No failure cases found in the evaluated samples.")
    
    # Add confusion matrix visualization
    st.markdown("## Advanced Analysis")
    
    # Calculate aggregated confusion matrix across all labels
    all_labels = list(VALID_LABELS)
    
    # Prepare for confusion matrix
    label_true_positive = {label: 0 for label in all_labels}
    label_false_positive = {label: 0 for label in all_labels}
    label_false_negative = {label: 0 for label in all_labels}
    label_true_negative = {label: 0 for label in all_labels}
    
    for true_labels, pred_labels in zip(true_labels_by_sample, ft_preds):
        for label in all_labels:
            if label in true_labels and label in pred_labels:
                label_true_positive[label] += 1
            elif label not in true_labels and label in pred_labels:
                label_false_positive[label] += 1
            elif label in true_labels and label not in pred_labels:
                label_false_negative[label] += 1
            else:
                label_true_negative[label] += 1
    
    # Create confusion matrix data
    def create_confusion_heatmap(label_data):
        """Create a confusion matrix heatmap for the given label data"""
        # Calculate percentages
        total_samples = sum(list(label_data.values()))
        labels = list(label_data.keys())
        matrix = np.zeros((2, 2))
        
        for i, label in enumerate(labels):
            true_pos = label_data[label]['tp']
            false_pos = label_data[label]['fp']
            false_neg = label_data[label]['fn']
            true_neg = label_data[label]['tn']
            
            # Calculate percentages
            matrix[0, 0] += true_pos / total_samples
            matrix[0, 1] += false_pos / total_samples
            matrix[1, 0] += false_neg / total_samples
            matrix[1, 1] += true_neg / total_samples
        
        # Create heatmap
        fig = px.imshow(
            matrix,
            labels=dict(x="Predicted", y="Actual", color="Percentage"),
            x=['Positive', 'Negative'],
            y=['Positive', 'Negative'],
            color_continuous_scale='RdBu_r',
            text_auto='.1%'
        )
        
        fig.update_layout(
            title='Aggregated Confusion Matrix',
            width=600,
            height=500
        )
        
        return fig
    
    # Create label data for heatmap
    label_data = {}
    for label in all_labels:
        label_data[label] = {
            'tp': label_true_positive[label],
            'fp': label_false_positive[label],
            'fn': label_false_negative[label],
            'tn': label_true_negative[label]
        }
    
    # Display confusion heatmap
    with st.expander("Confusion Matrix Analysis", expanded=False):
        st.markdown("### Per-Label Confusion Statistics")
        
        # Create confusion stats DataFrame
        confusion_df = pd.DataFrame({
            'Label': all_labels,
            'True Positive': [label_true_positive[label] for label in all_labels],
            'False Positive': [label_false_positive[label] for label in all_labels],
            'False Negative': [label_false_negative[label] for label in all_labels],
            'True Negative': [label_true_negative[label] for label in all_labels],
        })
        
        # Add precision, recall, and F1
        confusion_df['Precision'] = confusion_df.apply(
            lambda row: row['True Positive'] / max(1, row['True Positive'] + row['False Positive']),
            axis=1
        )
        
        confusion_df['Recall'] = confusion_df.apply(
            lambda row: row['True Positive'] / max(1, row['True Positive'] + row['False Negative']),
            axis=1
        )
        
        confusion_df['F1'] = confusion_df.apply(
            lambda row: 2 * row['Precision'] * row['Recall'] / max(0.001, row['Precision'] + row['Recall']),
            axis=1
        )
        
        # Format percentages
        for col in ['Precision', 'Recall', 'F1']:
            confusion_df[col] = confusion_df[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(confusion_df)
    
    # Add error analysis
    with st.expander("Error Analysis", expanded=False):
        st.markdown("### Common Error Patterns")
        
        # Find cases where model often confuses labels
        error_patterns = []
        
        for i, example in enumerate(examples):
            if set(example['true_labels']) != set(example['ft_prediction']):
                # Get missed and extra labels
                missed_labels = set(example['true_labels']) - set(example['ft_prediction'])
                extra_labels = set(example['ft_prediction']) - set(example['true_labels'])
                
                # Add to error patterns
                for missed in missed_labels:
                    for extra in extra_labels:
                        error_patterns.append((missed, extra))
        
        # Count error patterns
        error_counts = {}
        for pattern in error_patterns:
            if pattern in error_counts:
                error_counts[pattern] += 1
            else:
                error_counts[pattern] = 1
        
        # Create error pattern DataFrame
        if error_counts:
            error_df = pd.DataFrame({
                'True Label': [p[0] for p in error_counts.keys()],
                'Predicted Label': [p[1] for p in error_counts.keys()],
                'Count': list(error_counts.values())
            })
            
            # Sort by count
            error_df = error_df.sort_values('Count', ascending=False)
            
            st.markdown("#### Top Label Confusion Patterns")
            st.dataframe(error_df.head(10))
            
            # Create error pattern chart
            fig = px.bar(
                error_df.head(10),
                x='True Label',
                y='Count',
                color='Predicted Label',
                title='Top Label Confusion Patterns',
                labels={'x': 'True Label', 'y': 'Count', 'color': 'Predicted as'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error patterns found in the evaluated samples.")
    
    # Generate report section
    st.markdown("## Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate JSON download link
        export_data = {
            "metrics": {
                "base_model": base_metrics,
                "finetuned_model": ft_metrics
            },
            "temperature_analysis": results["finetuned_model"].get("by_temperature", {}),
            "best_temperature": results["finetuned_model"].get("best_temperature", None)
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        b64_json = base64.b64encode(json_str.encode()).decode()
        href_json = f'<a href="data:application/json;base64,{b64_json}" download="evaluation_results.json">📊 Download Evaluation Results (JSON)</a>'
        st.markdown(href_json, unsafe_allow_html=True)
    
    with col2:
        # Generate HTML report
        html_report = generate_html_report(results, examples)
        b64_html = base64.b64encode(html_report.encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="model_evaluation_report.html">📄 Download Full Evaluation Report (HTML)</a>'
        st.markdown(href_html, unsafe_allow_html=True)

def generate_html_report(results, examples):
    """Generate an enhanced HTML evaluation report"""
    # Extract metrics
    base_metrics = results["base_model"]["defaults"]
    ft_metrics = results["finetuned_model"]["defaults"]
    best_temp = results["finetuned_model"].get("best_temperature", None)
    
    # Create HTML report
    report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Parameter Model Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #1E88E5;
            }}
            h1 {{
                text-align: center;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .report-meta {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f8f8f8;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .metric-table th:first-child {{
                width: 30%;
            }}
            .example-container {{
                background-color: #f9f9f9;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
            }}
            .correct {{
                color: #4CAF50;
                font-weight: bold;
            }}
            .incorrect {{
                color: #F44336;
                font-weight: bold;
            }}
            .improvement {{
                color: #4CAF50;
            }}
            .regression {{
                color: #F44336;
            }}
            .chart-container {{
                margin: 30px 0;
                border: 1px solid #eee;
                padding: 20px;
                border-radius: 5px;
            }}
            footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #777;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <h1>Audio Parameter Model Evaluation Report</h1>
        
        <div class="report-meta">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Sample Count:</strong> {base_metrics['sample_count']}</p>
            <p><strong>Device:</strong> {"GPU" if torch.cuda.is_available() else "CPU"}</p>
            {f'<p><strong>Best Temperature:</strong> {best_temp["name"]} ({best_temp["temperature"]})</p>' if best_temp else ''}
        </div>
        
        <h2>Evaluation Summary</h2>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Base Model</th>
                <th>Fine-tuned Model</th>
                <th>Improvement</th>
            </tr>
    """
    
    # Add sample-level metrics
    for metric_name, metric_display in [('exact_match', 'Exact Match'), ('accuracy', 'Partial Match')]:
        base_value = base_metrics['sample_metrics'][metric_name]
        ft_value = ft_metrics['sample_metrics'][metric_name]
        change = ft_value - base_value
        change_pct = (change / base_value * 100) if base_value > 0 else float('inf')
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>{metric_display}</td>
                <td>{base_value:.4f}</td>
                <td>{ft_value:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
            </tr>
        """
    
    # Add micro metrics
    for metric_name, metric_display in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 Score')]:
        base_value = base_metrics['micro_metrics'][metric_name]
        ft_value = ft_metrics['micro_metrics'][metric_name]
        change = ft_value - base_value
        change_pct = (change / base_value * 100) if base_value > 0 else float('inf')
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>{metric_display} (Micro)</td>
                <td>{base_value:.4f}</td>
                <td>{ft_value:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
            </tr>
        """
    
    report += "</table>"
    
    # Add label-specific metrics
    report += """
        <h2>Label-specific F1 Scores</h2>
        <table>
            <tr>
                <th>Label</th>
                <th>Base Model</th>
                <th>Fine-tuned Model</th>
                <th>Improvement</th>
                <th>Support</th>
            </tr>
    """
    
    # Add label-level metrics
    for label in sorted(VALID_LABELS):
        base_f1 = base_metrics['label_metrics'][label]['f1']
        ft_f1 = ft_metrics['label_metrics'][label]['f1']
        change = ft_f1 - base_f1
        change_pct = (change / base_f1 * 100) if base_f1 > 0 else float('inf')
        support = base_metrics['label_metrics'][label]['support']
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>{label}</td>
                <td>{base_f1:.4f}</td>
                <td>{ft_f1:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
                <td>{support}</td>
            </tr>
        """
    
    report += "</table>"
    
    # Add temperature analysis if available
    if "by_temperature" in results["finetuned_model"] and results["finetuned_model"]["by_temperature"]:
        report += """
            <h2>Temperature Analysis</h2>
            <table>
                <tr>
                    <th>Temperature Name</th>
                    <th>Value</th>
                    <th>Exact Match</th>
                    <th>Partial Match</th>
                    <th>Micro F1</th>
                </tr>
        """
        
        # Sort by temperature value
        temp_configs = [(name, info) for name, info in results["finetuned_model"]["by_temperature"].items()]
        temp_configs.sort(key=lambda x: x[1]["temperature"])
        
        for temp_name, temp_info in temp_configs:
            temp_metrics = temp_info["metrics"]
            exact_match = temp_metrics["sample_metrics"]["exact_match"]
            partial_match = temp_metrics["sample_metrics"]["accuracy"]
            micro_f1 = temp_metrics["micro_metrics"]["f1"]
            
            # Highlight the best temperature
            is_best = best_temp and temp_name == best_temp["name"]
            row_style = 'style="background-color: #e3f2fd;"' if is_best else ''
            
            report += f"""
                <tr {row_style}>
                    <td>{temp_name}{" (Best)" if is_best else ""}</td>
                    <td>{temp_info["temperature"]}</td>
                    <td>{exact_match:.4f}</td>
                    <td>{partial_match:.4f}</td>
                    <td>{micro_f1:.4f}</td>
                </tr>
            """
        
        report += "</table>"
    
    # Add prediction examples
    report += """
        <h2>Prediction Examples</h2>
    """
    
    # Add success examples
    success_examples = [ex for ex in examples if set(ex['true_labels']) == set(ex['ft_prediction'])]
    if success_examples:
        report += "<h3>Success Cases</h3>"
        for i, example in enumerate(success_examples[:3]):
            report += f"""
            <div class="example-container">
                <p><strong>Input:</strong> {example['input']}</p>
                <p><strong>True Labels:</strong> {', '.join(example['true_labels'])}</p>
                <p><strong>Fine-tuned Model Prediction:</strong> <span class="correct">{', '.join(example['ft_prediction'])}</span></p>
            </div>
            """
    
    # Add improvement examples
    improvement_examples = [
        ex for ex in examples 
        if set(ex['true_labels']) == set(ex['ft_prediction']) and set(ex['true_labels']) != set(ex['base_prediction'])
    ]
    if improvement_examples:
        report += "<h3>Improvement Cases</h3>"
        for i, example in enumerate(improvement_examples[:3]):
            report += f"""
            <div class="example-container">
                <p><strong>Input:</strong> {example['input']}</p>
                <p><strong>True Labels:</strong> {', '.join(example['true_labels'])}</p>
                <p><strong>Base Model Prediction:</strong> <span class="incorrect">{', '.join(example['base_prediction'])}</span></p>
                <p><strong>Fine-tuned Model Prediction:</strong> <span class="correct">{', '.join(example['ft_prediction'])}</span></p>
            </div>
            """
    
    # Add failure examples
    failure_examples = [ex for ex in examples if set(ex['true_labels']) != set(ex['ft_prediction'])]
    if failure_examples:
        report += "<h3>Failure Cases</h3>"
        for i, example in enumerate(failure_examples[:3]):
            correct_labels = set(example['true_labels']) & set(example['ft_prediction'])
            missed_labels = set(example['true_labels']) - set(example['ft_prediction'])
            extra_labels = set(example['ft_prediction']) - set(example['true_labels'])
            
            report += f"""
            <div class="example-container">
                <p><strong>Input:</strong> {example['input']}</p>
                <p><strong>True Labels:</strong> {', '.join(example['true_labels'])}</p>
                <p><strong>Fine-tuned Model Prediction:</strong> <span class="incorrect">{', '.join(example['ft_prediction'])}</span></p>
            """
            
            if correct_labels:
                report += f'<p><strong>Correct Labels:</strong> <span class="correct">{", ".join(correct_labels)}</span></p>'
            if missed_labels:
                report += f'<p><strong>Missed Labels:</strong> {", ".join(missed_labels)}</p>'
            if extra_labels:
                report += f'<p><strong>Extra Labels:</strong> <span class="incorrect">{", ".join(extra_labels)}</span></p>'
            
            report += "</div>"
    
    # Add footer
    report += """
        <footer>
            <p>This report was automatically generated by the Audio Parameter Model Evaluation Tool</p>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </footer>
    </body>
    </html>
    """
    
    return report

# Add these functions to the main Streamlit application
def main():
    # Page config (keep original)
    st.set_page_config(
        page_title="Audio Parameter Model Evaluation",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply CSS styles (keep original styles)
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #333;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f5f5f5;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .metric-title {
            font-weight: bold;
            color: #555;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-comparison {
            font-size: 0.9rem;
            color: #4CAF50;
        }
        .negative-comparison {
            color: #F44336;
        }
        .example-container {
            background-color: #f9f9f9;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 4px 4px 0;
        }
        .correct-prediction {
            color: #4CAF50;
            font-weight: bold;
        }
        .incorrect-prediction {
            color: #F44336;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Entry point
if __name__ == "__main__":
    main()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select Function",
        ["Welcome", "Start Evaluation", "Dataset Analysis", "About"]
    )
    
    # Sidebar configuration
    st.sidebar.markdown("## Configuration")
    
    # Upload test data
    uploaded_file = st.sidebar.file_uploader("Upload Test Dataset", type=["json"])
    
    # Sample data
    if uploaded_file is None:
        st.sidebar.info("Please upload a JSON test dataset. Example data will be used if none is provided.")
        
        # Create example data
        test_data = []
        for i in range(100):
            # Randomly select 1-3 labels
            label_count = random.randint(1, 3)
            labels = random.sample(list(VALID_LABELS), label_count)
            
            # Create sample
            sample = {
                "instruction": "分析音频处理需求并选择相关参数（1-3个）：选项：低频/中频/高频/reverb/效果器/声场/压缩/音量",
                "input": f"Example input text {i+1}",
                "output": ", ".join(labels)
            }
            test_data.append(sample)
    else:
        try:
            test_data = json.load(uploaded_file)
            st.sidebar.success(f"Dataset loaded successfully with {len(test_data)} samples.")
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {str(e)}")
            test_data = []
    
    # Sample selection options
    with st.sidebar.expander("Sampling Options", expanded=False):
        # Sample ratio slider
        sample_ratio = st.slider("Percentage of data to use", 1, 100, 100)
        max_samples = int(len(test_data) * sample_ratio / 100)
        
        # Random seed
        seed = st.number_input("Random Seed", value=42, step=1)
        
        st.caption("Note: Changing the seed will result in different samples being selected for evaluation.")
    
    # Temperature settings
    with st.sidebar.expander("Temperature Settings", expanded=False):
        # Base model temperature
        base_temperature = st.slider(
            "Base Model Temperature",
            min_value=0.01,
            max_value=1.0,
            value=0.7,
            step=0.01,
            format="%.2f"
        )
        
        # Fine-tuned model temperature
        ft_temperature = st.slider(
            "Fine-tuned Model Temperature",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f"
        )
        
        # Temperature analysis checkbox
        run_temp_analysis = st.checkbox(
            "Run Temperature Analysis", 
            value=False, 
            help="Test multiple temperature settings to find the optimal value"
        )
    
    # Model path configuration
    with st.sidebar.expander("Model Path Configuration", expanded=False):
        base_model_path = st.text_input(
            "Base Model Path",
            value="../../models/base_model"
        )
        
        ft_model_path = st.text_input(
            "Fine-tuned Model Path",
            value="../../models/finetuned_model"
        )
    
    # Create model paths dictionary
    model_paths = {
        "base": base_model_path,
        "finetuned": ft_model_path
    }
    
    # Page content
    if page == "Welcome":
        # Welcome page
        st.markdown('<h1 class="main-header">Audio Parameter Model Evaluation Tool</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.image("https://img.icons8.com/fluency/96/000000/musical-notes.png", width=80)
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #f5f9ff; border-radius: 10px; margin: 20px 0;">
                <h2 style="margin-top: 10px;">Welcome to the Enhanced Model Evaluation Tool</h2>
                <p>This tool helps you compare the performance of base and fine-tuned models on audio parameter classification tasks.</p>
                <p>The tool incorporates best practices from both existing implementations to provide comprehensive evaluation capabilities.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Key Features
        
        - **Enhanced Model Evaluation**: Test models with optimal evaluation processes from both implementations
        - **Temperature Analysis**: Find the optimal temperature setting for your model
        - **Detailed Label Analysis**: Deep insights into per-label performance
        - **Visualizations**: Interactive charts and graphs to understand model performance
        - **Comprehensive Reports**: Export detailed HTML and JSON reports
        
        ### How to Use
        
        1. Select "Start Evaluation" in the sidebar
        2. Upload a test dataset (JSON format)
        3. Configure evaluation parameters
        4. Click the "Start Evaluation" button
        5. View evaluation results and visualizations
        6. Export evaluation report
        """)
        
        # Show quick start guide
        with st.expander("Quick Start Guide", expanded=False):
            st.markdown("""
            #### Dataset Format
            
            The tool expects a JSON dataset with the following structure:
            
            ```json
            [
                {
                    "instruction": "Prompt instruction",
                    "input": "Audio description text",
                    "output": "low frequency, reverb"
                },
                ...
            ]
            ```
            
            #### Temperature Analysis
            
            The temperature setting controls how deterministic (lower values) or creative (higher values) the model's outputs are:
            
            - **Ultra-low (0.01)**: Most deterministic, same outputs every time
            - **Low (0.1)**: Minimal variation, good for consistent outputs
            - **Medium (0.3)**: Some variation, balanced approach
            - **High (0.7)**: More variation, creative outputs
            - **Ultra-high (1.0)**: Maximum variation, most creative
            
            For classification tasks, lower temperatures often perform better.
            """)
    
    elif page == "Start Evaluation":
        st.markdown('<h1 class="main-header">Model Evaluation</h1>', unsafe_allow_html=True)
        
        if len(test_data) == 0:
            st.warning("Please upload a test dataset or use example data.")
            st.stop()
        
        # Evaluation button
        if st.button("🚀 Start Evaluation", key="run_evaluation"):
            with st.spinner("Evaluating models..."):
                # Configure evaluation settings
                eval_config = {
                    "run_temp_analysis": run_temp_analysis,
                    "base_temperature": base_temperature,
                    "ft_temperature": ft_temperature,
                    "max_samples": max_samples,
                    "seed": seed
                }
                
                # Run optimized evaluation
                results, examples = run_optimized_evaluation(test_data, model_paths, max_samples, seed)
                
                if results and examples:
                    # Save results to session state
                    st.session_state.results = results
                    st.session_state.examples = examples
                    st.session_state.true_labels_by_sample = [ex['true_labels'] for ex in examples]
                    st.session_state.ft_preds = [ex['ft_prediction'] for ex in examples]
                    st.session_state.config = eval_config
                else:
                    st.error("Evaluation failed. Please check model paths and data format.")
        
        # Display evaluation results

        if 'results' in st.session_state and 'examples' in st.session_state:
            display_optimized_results(
                st.session_state.results, 
                st.session_state.examples,
                st.session_state.get('true_labels_by_sample'),
                st.session_state.get('ft_preds')
            )
    
    elif page == "Dataset Analysis":
        st.markdown('<h1 class="main-header">Dataset Analysis</h1>', unsafe_allow_html=True)
        
        if len(test_data) == 0:
            st.warning("Please upload a test dataset or use example data.")
            st.stop()
        
        # Extract and calculate label distribution
        all_labels = []
        for item in test_data:
            labels = parse_labels(item.get('output', ''))
            all_labels.extend(labels)
        
        # Label distribution
        label_counts = {}
        for label in VALID_LABELS:
            label_counts[label] = all_labels.count(label)
        
        # Multi-label distribution
        multi_label_counts = {}
        for item in test_data:
            labels = parse_labels(item.get('output', ''))
            count = len(labels)
            multi_label_counts[count] = multi_label_counts.get(count, 0) + 1
        
        # Display basic statistics
        st.markdown("### Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(test_data))
        
        with col2:
            avg_labels = len(all_labels) / max(1, len(test_data))
            st.metric("Avg. Labels per Sample", f"{avg_labels:.2f}")
        
        with col3:
            unique_labels = len(set(all_labels))
            st.metric("Unique Labels Used", f"{unique_labels}/{len(VALID_LABELS)}")
        
        # Label distribution chart
        st.markdown("### Label Distribution")
        
        label_df = pd.DataFrame({
            'Label': list(label_counts.keys()),
            'Count': list(label_counts.values())
        })
        
        label_df = label_df.sort_values('Count', ascending=False)
        
        fig = px.bar(
            label_df,
            x='Label',
            y='Count',
            title='Label Distribution',
            color='Count',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-label distribution chart
        st.markdown("### Multi-label Distribution")
        
        multi_label_df = pd.DataFrame({
            'Label Count': list(multi_label_counts.keys()),
            'Sample Count': list(multi_label_counts.values())
        })
        
        multi_label_df = multi_label_df.sort_values('Label Count')
        
        fig = px.bar(
            multi_label_df,
            x='Label Count',
            y='Sample Count',
            title='Multi-label Distribution',
            color='Sample Count',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Label co-occurrence analysis
        st.markdown("### Label Co-occurrence Analysis")
        
        # Calculate co-occurrence matrix
        cooccurrence = np.zeros((len(VALID_LABELS), len(VALID_LABELS)))
        labels_list = list(VALID_LABELS)
        
        for item in test_data:
            item_labels = parse_labels(item.get('output', ''))
            for i, label1 in enumerate(labels_list):
                for j, label2 in enumerate(labels_list):
                    if label1 in item_labels and label2 in item_labels:
                        cooccurrence[i, j] += 1
        
        # Create co-occurrence heatmap
        fig = px.imshow(
            cooccurrence,
            labels=dict(x="Label", y="Label", color="Co-occurrence"),
            x=labels_list,
            y=labels_list,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title='Label Co-occurrence Matrix',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "About":
        st.markdown('<h1 class="main-header">About This Tool</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Enhanced Audio Parameter Model Evaluation Tool
        
        This tool combines the best features from multiple evaluation implementations to provide a comprehensive evaluation suite for audio parameter classification models.
        
        #### Key Optimizations
        
        1. **Enhanced Label Parsing**: Improved handling of both English and Chinese labels with more robust parsing
        2. **Temperature Analysis**: Testing models with different temperature settings to find the optimal configuration
        3. **Advanced Metrics**: Comprehensive metrics including per-label analysis and confusion patterns
        4. **Interactive Visualizations**: Added interactive charts and tables for better data exploration
        5. **Error Analysis**: Detailed error pattern analysis to identify model weaknesses
        
        #### Deployment Instructions
        
        To deploy this tool:
        
        1. Install required dependencies:
           ```
           pip install streamlit torch transformers pandas numpy plotly scikit-learn seaborn tqdm
           ```
           
        2. Run the application:
           ```
           streamlit run app.py
           ```
           
        3. Access the web interface at http://localhost:8501
        """)
        
        # Show technical details
        with st.expander("Technical Implementation Details", expanded=False):
            st.markdown("""
            #### Model Evaluation Process
            
            The evaluation process follows these steps:
            
            1. Load models and tokenizers using Hugging Face Transformers
            2. Process test samples to extract inputs and reference labels
            3. Generate predictions with both base and fine-tuned models
            4. Calculate metrics for both models and compare performance
            5. Optionally test different temperature settings
            6. Generate comprehensive evaluation report
            
            #### Label Parsing Logic
            
            The label parsing logic combines exact matching, fuzzy matching, and fallback mechanisms:
            
            1. Try exact matching against valid labels list
            2. Try matching against known synonyms and translations
            3. Apply fuzzy matching based on substrings
            4. Use fallback logic to ensure at least one label is returned
            
            #### Metrics Calculation
            
            The metrics calculation involves:
            
            1. Sample-level metrics (exact match, partial match)
            2. Micro-averaged metrics across all labels
            3. Per-label precision, recall, and F1 scores
            4. Error pattern analysis and confusion detection
            """)
    
    # Footer
    st.markdown("""
    <footer style="text-align: center; margin-top: 2rem; padding: 1rem; color: #777; font-size: 0.9rem;">
        <p>Audio Parameter Model Evaluation Tool | Version 2.0 | MixMaster Team</p>
    </footer>
    """, unsafe_allow_html=True)