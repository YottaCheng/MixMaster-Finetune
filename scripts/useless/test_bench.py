import streamlit as st
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
from datetime import datetime
import uuid
import base64
import io

# 配置页面
st.set_page_config(
    page_title="音频参数模型评估工具",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
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
    .tab-container {
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-top: 10px;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: #777;
        font-size: 0.9rem;
    }
    .prediction-example {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
    }
    .prediction-text {
        font-size: 1.1rem;
        color: #444;
    }
    .correct-prediction {
        color: #4CAF50;
        font-weight: bold;
    }
    .incorrect-prediction {
        color: #F44336;
        font-weight: bold;
    }
    .custom-progress {
        height: 20px;
        border-radius: 10px;
    }
    .help-text {
        color: #777;
        font-size: 0.9rem;
        font-style: italic;
    }
    .comparison-highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 常量定义
VALID_LABELS = {'低频', '中频', '高频', 'reverb', '效果器', '声场', '压缩', '音量'}
DEFAULT_MAX_SAMPLES = 50
DEFAULT_SEED = 42

# 工具函数
def set_seed(seed):
    """设置随机种子确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_labels(text):
    """解析并标准化标签"""
    if not text or not isinstance(text, str):
        return []
        
    text = text.strip().lower()
    # 处理中英文标点和空格
    raw_labels = [label.strip() for label in re.split(r'[,，、]', text) if label.strip()]
    
    # 验证标签并检测乱码
    validated_labels = []
    for label in raw_labels:
        # 检测乱码
        if re.search(r'[\ufffd\uFFFD]', label):
            # 尝试修复常见乱码
            if '压' in label and '缩' in label:
                validated_labels.append('压缩')
            else:
                st.warning(f"检测到乱码标签 '{label}'")
        else:
            # 规范化标签
            if label in VALID_LABELS:
                validated_labels.append(label)
            # 处理近似匹配
            elif '低' in label and '频' in label:
                validated_labels.append('低频')
            elif '中' in label and '频' in label:
                validated_labels.append('中频')
            elif '高' in label and '频' in label:
                validated_labels.append('高频')
            elif 'reverb' in label or '混响' in label:
                validated_labels.append('reverb')
            elif '效果' in label:
                validated_labels.append('效果器')
            elif '声场' in label:
                validated_labels.append('声场')
            elif '压' in label and '缩' in label:
                validated_labels.append('压缩')
            elif '音量' in label or '音量' in label:
                validated_labels.append('音量')
            
    return validated_labels

def extract_prediction(text, prompt):
    """从模型生成的文本中提取预测标签"""
    # 移除prompt部分
    if prompt in text:
        text = text[len(prompt):]
    
    # 尝试查找"输出："部分
    if "输出：" in text:
        text = text.split("输出：")[1].strip()
    
    # 清理文本
    text = re.sub(r'^[\s:"：]+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    
    # 解析标签
    return parse_labels(text)

def get_download_link(data, filename, text):
    """生成下载链接"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    else:
        b64 = base64.b64encode(json.dumps(data, indent=2, ensure_ascii=False).encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def generate_html_report(metrics, examples, config):
    """生成HTML评估报告"""
    base_metrics = metrics["base_model"]
    ft_metrics = metrics["finetuned_model"]
    
    # 创建HTML报告
    report = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音频参数模型评估报告</title>
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
        <h1>音频参数模型评估报告</h1>
        
        <div class="report-meta">
            <p><strong>生成时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>评估配置：</strong></p>
            <ul>
                <li>随机种子: {config.get('seed', 'N/A')}</li>
                <li>评估样本数: {config.get('sample_count', 'N/A')}</li>
                <li>运行设备: {config.get('device', 'N/A')}</li>
            </ul>
        </div>
        
        <h2>评估摘要</h2>
        <table class="metric-table">
            <tr>
                <th>指标</th>
                <th>基础模型</th>
                <th>微调模型</th>
                <th>提升</th>
            </tr>
    """
    
    # 添加样本级指标
    for metric in ['accuracy', 'exact_match']:
        base_value = base_metrics['sample_metrics'][metric]
        ft_value = ft_metrics['sample_metrics'][metric]
        change = ft_value - base_value
        change_pct = (change / base_value * 100) if base_value > 0 else float('inf')
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td>{base_value:.4f}</td>
                <td>{ft_value:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
            </tr>
        """
    
    # 添加微观指标
    for metric in ['precision', 'recall', 'f1']:
        base_value = base_metrics['micro_metrics'][metric]
        ft_value = ft_metrics['micro_metrics'][metric]
        change = ft_value - base_value
        change_pct = (change / base_value * 100) if base_value > 0 else float('inf')
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>Micro-{metric.title()}</td>
                <td>{base_value:.4f}</td>
                <td>{ft_value:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
            </tr>
        """
    
    # 样本数量
    report += f"""
            <tr>
                <td>样本数量</td>
                <td>{base_metrics['sample_count']}</td>
                <td>{ft_metrics['sample_count']}</td>
                <td>-</td>
            </tr>
        </table>
        
        <h2>各标签F1分数</h2>
        <table>
            <tr>
                <th>标签</th>
                <th>基础模型</th>
                <th>微调模型</th>
                <th>提升</th>
            </tr>
    """
    
    # 添加标签级指标
    for label in sorted(VALID_LABELS):
        base_f1 = base_metrics['label_metrics'][label]['f1']
        ft_f1 = ft_metrics['label_metrics'][label]['f1']
        change = ft_f1 - base_f1
        change_pct = (change / base_f1 * 100) if base_f1 > 0 else float('inf')
        
        css_class = "improvement" if change > 0 else "regression"
        report += f"""
            <tr>
                <td>{label}</td>
                <td>{base_f1:.4f}</td>
                <td>{ft_f1:.4f}</td>
                <td class="{css_class}">{change_pct:+.2f}%</td>
            </tr>
        """
    
    report += "</table>"
    
    # 添加预测示例
    report += """
        <h2>预测示例</h2>
    """
    
    for i, example in enumerate(examples[:5]):
        input_text = example['input']
        true_labels = example['true_labels']
        base_pred = example['base_prediction']
        ft_pred = example['ft_prediction']
        
        base_correct = set(true_labels) == set(base_pred)
        ft_correct = set(true_labels) == set(ft_pred)
        
        base_class = "correct" if base_correct else "incorrect"
        ft_class = "correct" if ft_correct else "incorrect"
        
        report += f"""
        <div class="example-container">
            <h3>示例 {i+1}</h3>
            <p><strong>输入:</strong> {input_text}</p>
            <p><strong>真实标签:</strong> {', '.join(true_labels)}</p>
            <p><strong>基础模型预测:</strong> <span class="{base_class}">{', '.join(base_pred)}</span></p>
            <p><strong>微调模型预测:</strong> <span class="{ft_class}">{', '.join(ft_pred)}</span></p>
        </div>
        """
    
    # 添加页脚
    report += """
        <footer>
            <p>本报告由音频参数模型评估工具自动生成</p>
        </footer>
    </body>
    </html>
    """
    
    return report

# 数据处理功能
def preprocess_test_data(data, sample_ratio=100, seed=42):
    """预处理测试数据"""
    # 设置随机种子
    random.seed(seed)
    
    # 验证数据格式
    valid_data = []
    for idx, item in enumerate(data):
        if not all(k in item for k in ['instruction', 'input', 'output']):
            continue
            
        if not item['output'].strip():
            continue
            
        valid_data.append(item)
    
    # 计算采样数量
    sample_count = max(1, int(len(valid_data) * sample_ratio / 100))
    
    # 随机采样
    if sample_count < len(valid_data):
        valid_data = random.sample(valid_data, sample_count)
    
    return valid_data

def create_evaluation_metrics(all_true_labels, all_pred_labels, label_set):
    """计算多标签评估指标"""
    # 构建二值化数组
    y_true = []
    y_pred = []
    
    for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
        true_vec = [1 if label in true_labels else 0 for label in label_set]
        pred_vec = [1 if label in pred_labels else 0 for label in label_set]
        y_true.append(true_vec)
        y_pred.append(pred_vec)
    
    # 计算标签级指标
    label_metrics = {}
    for i, label in enumerate(label_set):
        true_label = [row[i] for row in y_true]
        pred_label = [row[i] for row in y_pred]
        
        # 避免除零错误
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
    
    # 样本级指标
    sample_accuracy = sum(1 for t, p in zip(all_true_labels, all_pred_labels) 
                          if len(set(t) & set(p)) > 0) / max(1, len(all_true_labels))
    
    exact_match = sum(1 for t, p in zip(all_true_labels, all_pred_labels) 
                     if set(t) == set(p)) / max(1, len(all_true_labels))
    
    # 标签级指标
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

# 界面功能
def show_welcome():
    """显示欢迎页面"""
    st.markdown('<h1 class="main-header">音频参数模型评估工具</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f5f9ff; border-radius: 10px; margin: 20px 0;">
            <img src="https://img.icons8.com/fluency/96/000000/musical-notes.png" style="width: 80px;">
            <h2 style="margin-top: 10px;">欢迎使用模型评估工具</h2>
            <p>这个工具可以帮助您比较基础模型和微调模型在音频参数分类任务上的表现。</p>
            <p>请在左侧导航栏中选择要使用的功能。</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 主要功能
    
    - **模型评估**：上传测试数据集，评估两个模型的性能
    - **数据集分析**：查看数据集的分布和统计信息
    - **结果可视化**：通过图表直观比较模型性能
    - **生成报告**：导出详细的评估报告
    
    ### 使用步骤
    
    1. 在侧边栏中选择"开始评估"
    2. 上传测试数据集（JSON格式）
    3. 配置评估参数
    4. 点击"开始评估"按钮
    5. 查看评估结果和可视化
    6. 导出评估报告
    """)

def run_evaluation(test_data, prompts, max_samples=50, seed=42):
    """运行模型评估"""
    # 示例评估（在实际场景中会调用预训练模型和微调模型）
    # 这里使用模拟数据展示界面
    
    # 设置进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 模拟加载模型
    status_text.text("正在加载模型...")
    time.sleep(1)
    progress_bar.progress(10)
    
    # 准备测试数据
    status_text.text("准备测试数据...")
    set_seed(seed)
    test_samples = preprocess_test_data(test_data, sample_ratio=100, seed=seed)
    test_samples = test_samples[:max_samples]  # 限制样本数
    time.sleep(1)
    progress_bar.progress(20)
    
    # 模拟评估过程
    all_true_labels = []
    base_pred_labels = []
    ft_pred_labels = []
    examples = []
    
    total_samples = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        # 更新进度
        progress = 20 + int(70 * (i+1) / total_samples)
        progress_bar.progress(progress)
        status_text.text(f"评估样本 {i+1}/{total_samples}...")
        
        # 提取真实标签
        true_labels = parse_labels(sample['output'])
        
        # 在实际场景中，这里会调用模型生成预测
        # 这里使用模拟数据
        
        # 模拟基础模型预测（随机性较高）
        base_correct = random.random() < 0.4  # 40%正确率
        if base_correct:
            base_pred = true_labels
        else:
            # 随机选择1-3个标签
            available_labels = list(VALID_LABELS - set(true_labels))
            if available_labels:
                wrong_count = min(random.randint(1, 3), len(available_labels))
                wrong_labels = random.sample(available_labels, wrong_count)
                base_pred = wrong_labels
            else:
                base_pred = []
        
        # 模拟微调模型预测（更高的正确率）
        ft_correct = random.random() < 0.75  # 75%正确率
        if ft_correct:
            ft_pred = true_labels
        else:
            # 随机选择1-2个标签，并有50%概率保留部分正确标签
            available_labels = list(VALID_LABELS - set(true_labels))
            if available_labels:
                wrong_count = min(random.randint(1, 2), len(available_labels))
                wrong_labels = random.sample(available_labels, wrong_count)
                
                # 有50%几率保留部分正确标签
                if true_labels and random.random() < 0.5:
                    keep_count = random.randint(1, len(true_labels))
                    kept_true = random.sample(true_labels, keep_count)
                    ft_pred = wrong_labels + kept_true
                else:
                    ft_pred = wrong_labels
            else:
                ft_pred = []
        
        # 收集结果
        all_true_labels.append(true_labels)
        base_pred_labels.append(base_pred)
        ft_pred_labels.append(ft_pred)
        
        # 收集示例
        examples.append({
            'input': sample['input'],
            'true_labels': true_labels,
            'base_prediction': base_pred,
            'ft_prediction': ft_pred
        })
        
        # 模拟处理时间
        time.sleep(0.01)
    
    # 计算指标
    status_text.text("计算评估指标...")
    progress_bar.progress(95)
    
    base_metrics = create_evaluation_metrics(all_true_labels, base_pred_labels, list(VALID_LABELS))
    ft_metrics = create_evaluation_metrics(all_true_labels, ft_pred_labels, list(VALID_LABELS))
    
    # 完成评估
    progress_bar.progress(100)
    status_text.text("评估完成！")
    time.sleep(0.5)
    status_text.empty()
    
    return {
        "base_model": base_metrics,
        "finetuned_model": ft_metrics
    }, examples

def display_evaluation_results(metrics, examples):
    """显示评估结果"""
    base_metrics = metrics["base_model"]
    ft_metrics = metrics["finetuned_model"]
    
    st.markdown('<h2 class="sub-header">评估结果摘要</h2>', unsafe_allow_html=True)
    
    # 主要指标卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">准确率</div>', unsafe_allow_html=True)
        
        base_acc = base_metrics['sample_metrics']['accuracy']
        ft_acc = ft_metrics['sample_metrics']['accuracy']
        change = ft_acc - base_acc
        change_pct = (change / base_acc * 100) if base_acc > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_acc:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">与基础模型相比: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">F1分数 (微平均)</div>', unsafe_allow_html=True)
        
        base_f1 = base_metrics['micro_metrics']['f1']
        ft_f1 = ft_metrics['micro_metrics']['f1']
        change = ft_f1 - base_f1
        change_pct = (change / base_f1 * 100) if base_f1 > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_f1:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">与基础模型相比: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">完全匹配率</div>', unsafe_allow_html=True)
        
        base_exact = base_metrics['sample_metrics']['exact_match']
        ft_exact = ft_metrics['sample_metrics']['exact_match']
        change = ft_exact - base_exact
        change_pct = (change / base_exact * 100) if base_exact > 0 else float('inf')
        
        st.markdown(f'<div class="metric-value">{ft_exact:.2%}</div>', unsafe_allow_html=True)
        
        comparison_class = "metric-comparison" if change >= 0 else "metric-comparison negative-comparison"
        st.markdown(f'<div class="{comparison_class}">与基础模型相比: {change_pct:+.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 详细指标表格
    st.markdown('<h3 class="sub-header">详细评估指标</h3>', unsafe_allow_html=True)
    
# 创建DataFrame
    metrics_df = pd.DataFrame({
        '指标': ['准确率', '完全匹配率', '微平均精确率', '微平均召回率', '微平均F1', '宏平均精确率', '宏平均召回率', '宏平均F1'],
        '基础模型': [
            base_metrics['sample_metrics']['accuracy'],
            base_metrics['sample_metrics']['exact_match'],
            base_metrics['micro_metrics']['precision'],
            base_metrics['micro_metrics']['recall'],
            base_metrics['micro_metrics']['f1'],
            base_metrics['macro_metrics']['precision'],
            base_metrics['macro_metrics']['recall'],
            base_metrics['macro_metrics']['f1']
        ],
        '微调模型': [
            ft_metrics['sample_metrics']['accuracy'],
            ft_metrics['sample_metrics']['exact_match'],
            ft_metrics['micro_metrics']['precision'],
            ft_metrics['micro_metrics']['recall'],
            ft_metrics['micro_metrics']['f1'],
            ft_metrics['macro_metrics']['precision'],
            ft_metrics['macro_metrics']['recall'],
            ft_metrics['macro_metrics']['f1']
        ]
    })
    
    # 计算提升
    metrics_df['提升'] = metrics_df['微调模型'] - metrics_df['基础模型']
    metrics_df['提升百分比'] = metrics_df.apply(
        lambda row: ((row['微调模型'] - row['基础模型']) / row['基础模型'] * 100) if row['基础模型'] > 0 else float('inf'),
        axis=1
    )
    
    # 格式化为百分比
    format_cols = ['基础模型', '微调模型', '提升']
    metrics_df[format_cols] = metrics_df[format_cols].applymap(lambda x: f"{x:.2%}")
    metrics_df['提升百分比'] = metrics_df['提升百分比'].apply(lambda x: f"{x:+.2f}%" if x != float('inf') else "N/A")
    
    st.dataframe(metrics_df)
    
    # 绘制对比图表
    st.markdown('<h3 class="sub-header">性能对比</h3>', unsafe_allow_html=True)
    
    fig = go.Figure()
    metrics_to_plot = ['准确率', '完全匹配率', '微平均F1', '宏平均F1']
    
    base_values = [float(row['基础模型'].strip('%'))/100 for _, row in metrics_df[metrics_df['指标'].isin(metrics_to_plot)].iterrows()]
    ft_values = [float(row['微调模型'].strip('%'))/100 for _, row in metrics_df[metrics_df['指标'].isin(metrics_to_plot)].iterrows()]
    
    fig.add_trace(go.Bar(
        x=metrics_to_plot,
        y=base_values,
        name='基础模型',
        marker_color='#90CAF9'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics_to_plot,
        y=ft_values,
        name='微调模型',
        marker_color='#F48FB1'
    ))
    
    fig.update_layout(
        barmode='group',
        title='模型性能对比',
        yaxis=dict(
            title='得分',
            tickformat='.0%'
        ),
        xaxis=dict(
            title='评估指标'
        ),
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 标签级性能
    st.markdown('<h3 class="sub-header">各标签性能</h3>', unsafe_allow_html=True)
    
    label_df = pd.DataFrame({
        '标签': list(VALID_LABELS),
        '基础模型 F1': [base_metrics['label_metrics'][label]['f1'] for label in VALID_LABELS],
        '微调模型 F1': [ft_metrics['label_metrics'][label]['f1'] for label in VALID_LABELS],
        '样本数量': [base_metrics['label_metrics'][label]['support'] for label in VALID_LABELS]
    })
    
    label_df['提升'] = label_df['微调模型 F1'] - label_df['基础模型 F1']
    label_df['提升百分比'] = label_df.apply(
        lambda row: ((row['微调模型 F1'] - row['基础模型 F1']) / row['基础模型 F1'] * 100) if row['基础模型 F1'] > 0 else float('inf'),
        axis=1
    )
    
    # 格式化为百分比
    format_cols = ['基础模型 F1', '微调模型 F1']
    label_df[format_cols] = label_df[format_cols].applymap(lambda x: f"{x:.2%}")
    label_df['提升'] = label_df['提升'].apply(lambda x: f"{x:.2%}")
    label_df['提升百分比'] = label_df['提升百分比'].apply(lambda x: f"{x:+.2f}%" if x != float('inf') else "N/A")
    
    st.dataframe(label_df)
    
    # 绘制标签级性能对比图
    fig = px.bar(
        label_df,
        x='标签',
        y=['基础模型 F1', '微调模型 F1'],
        barmode='group',
        labels={'value': 'F1分数', 'variable': '模型'},
        title='各标签F1分数对比',
        color_discrete_map={
            '基础模型 F1': '#90CAF9',
            '微调模型 F1': '#F48FB1'
        }
    )
    
    fig.update_layout(
        xaxis=dict(title='标签'),
        yaxis=dict(title='F1分数', tickformat='.0%'),
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 预测示例
    st.markdown('<h3 class="sub-header">预测示例</h3>', unsafe_allow_html=True)
    
    for i, example in enumerate(examples[:5]):
        with st.expander(f"示例 {i+1}: {example['input'][:50]}...", expanded=(i==0)):
            st.markdown(f"**输入文本**: {example['input']}")
            st.markdown(f"**真实标签**: {', '.join(example['true_labels'])}")
            
            base_correct = set(example['true_labels']) == set(example['base_prediction'])
            ft_correct = set(example['true_labels']) == set(example['ft_prediction'])
            
            base_class = "correct-prediction" if base_correct else "incorrect-prediction"
            ft_class = "correct-prediction" if ft_correct else "incorrect-prediction"
            
            st.markdown(f"**基础模型预测**: <span class='{base_class}'>{', '.join(example['base_prediction'])}</span>", unsafe_allow_html=True)
            st.markdown(f"**微调模型预测**: <span class='{ft_class}'>{', '.join(example['ft_prediction'])}</span>", unsafe_allow_html=True)

def display_data_analysis(test_data):
    """显示数据集分析"""
    st.markdown('<h2 class="sub-header">数据集分析</h2>', unsafe_allow_html=True)
    
    # 基本统计信息
    st.markdown("### 基本统计信息")
    
    total_samples = len(test_data)
    
    # 提取并计算标签分布
    all_labels = []
    for item in test_data:
        labels = parse_labels(item.get('output', ''))
        all_labels.extend(labels)
    
    # 标签分布
    label_counts = {}
    for label in VALID_LABELS:
        label_counts[label] = all_labels.count(label)
    
    # 多标签分布
    multi_label_counts = {}
    for item in test_data:
        labels = parse_labels(item.get('output', ''))
        count = len(labels)
        multi_label_counts[count] = multi_label_counts.get(count, 0) + 1
    
    # 显示基本统计信息
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("样本总数", total_samples)
    
    with col2:
        avg_labels = len(all_labels) / max(1, total_samples)
        st.metric("平均标签数", f"{avg_labels:.2f}")
    
    with col3:
        unique_labels = len(set(all_labels))
        st.metric("使用的不同标签数", f"{unique_labels}/{len(VALID_LABELS)}")
    
    # 标签分布图
    st.markdown("### 标签分布")
    
    label_df = pd.DataFrame({
        '标签': list(label_counts.keys()),
        '出现次数': list(label_counts.values())
    })
    
    label_df = label_df.sort_values('出现次数', ascending=False)
    
    fig = px.bar(
        label_df,
        x='标签',
        y='出现次数',
        title='标签分布',
        color='出现次数',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis=dict(title='标签'),
        yaxis=dict(title='出现次数'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 多标签分布图
    st.markdown("### 多标签分布")
    
    multi_label_df = pd.DataFrame({
        '标签数量': list(multi_label_counts.keys()),
        '样本数': list(multi_label_counts.values())
    })
    
    multi_label_df = multi_label_df.sort_values('标签数量')
    
    fig = px.bar(
        multi_label_df,
        x='标签数量',
        y='样本数',
        title='多标签分布',
        color='样本数',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis=dict(title='每个样本的标签数量'),
        yaxis=dict(title='样本数'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 标签共现矩阵
    st.markdown("### 标签共现矩阵")
    
    # 创建共现矩阵
    cooccurrence = np.zeros((len(VALID_LABELS), len(VALID_LABELS)))
    valid_labels_list = list(VALID_LABELS)
    
    for item in test_data:
        labels = parse_labels(item.get('output', ''))
        for i, label1 in enumerate(valid_labels_list):
            for j, label2 in enumerate(valid_labels_list):
                if label1 in labels and label2 in labels:
                    cooccurrence[i, j] += 1
    
    # 绘制热图
    fig = px.imshow(
        cooccurrence,
        x=valid_labels_list,
        y=valid_labels_list,
        color_continuous_scale='Viridis',
        title='标签共现矩阵'
    )
    
    fig.update_layout(
        height=600,
        width=600
    )
    
    st.plotly_chart(fig)
    
    # 显示数据样本
    st.markdown("### 数据样本预览")
    
    samples = test_data[:10]  # 限制显示前10个样本
    for i, sample in enumerate(samples):
        with st.expander(f"样本 {i+1}", expanded=(i==0)):
            st.markdown(f"**指令**: {sample.get('instruction', 'N/A')}")
            st.markdown(f"**输入**: {sample.get('input', 'N/A')}")
            st.markdown(f"**输出**: {sample.get('output', 'N/A')}")
            
            # 解析标签
            labels = parse_labels(sample.get('output', ''))
            st.markdown(f"**解析后的标签**: {', '.join(labels)}")

def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.markdown("## 导航")
    page = st.sidebar.radio(
        "选择功能",
        ["欢迎页面", "开始评估", "数据集分析"]
    )
    
    # 侧边栏配置
    st.sidebar.markdown("## 配置")
    
    # 上传测试数据
    uploaded_file = st.sidebar.file_uploader("上传测试数据集", type=["json"])
    
    # 示例数据
    if uploaded_file is None:
        st.sidebar.info("请上传JSON格式的测试数据集。如果没有数据集，将使用示例数据。")
        # 创建示例数据
        test_data = []
        for i in range(100):
            # 随机选择1-3个标签
            label_count = random.randint(1, 3)
            labels = random.sample(list(VALID_LABELS), label_count)
            
            # 创建样本
            sample = {
                "instruction": "分析音频处理需求并选择相关参数（1-3个）：选项：低频/中频/高频/reverb/效果器/声场/压缩/音量",
                "input": f"示例输入文本 {i+1}",
                "output": ", ".join(labels)
            }
            test_data.append(sample)
    else:
        try:
            test_data = json.load(uploaded_file)
            st.sidebar.success(f"成功加载数据集，包含 {len(test_data)} 个样本。")
        except Exception as e:
            st.sidebar.error(f"加载数据失败: {str(e)}")
            test_data = []
    
    # 其他配置
    max_samples = st.sidebar.slider("评估样本数量", 10, 200, DEFAULT_MAX_SAMPLES)
    sample_ratio = st.sidebar.slider("样本比例 (%)", 1, 100, 100)
    seed = st.sidebar.number_input("随机种子", value=DEFAULT_SEED)
    
    # 模型提示词配置
    with st.sidebar.expander("提示词配置", expanded=False):
        base_prompt = st.text_area(
            "基础模型提示词",
            value="输入：[TEXT]\n输出："
        )
        
        ft_prompt = st.text_area(
            "微调模型提示词",
            value="分析音频处理需求并选择相关参数（1-3个）：\n\n输入：[TEXT]\n输出："
        )
    
    # 模型路径配置
    with st.sidebar.expander("模型路径配置", expanded=False):
        base_model_path = st.text_input(
            "基础模型路径",
            value="../../models/base_model"
        )
        
        ft_model_path = st.text_input(
            "微调模型路径",
            value="../../models/finetuned_model"
        )
    
    # 创建提示词字典
    prompts = {
        "base": base_prompt,
        "finetuned": ft_prompt
    }
    
    # 页面内容
    if page == "欢迎页面":
        show_welcome()
    
    elif page == "开始评估":
        st.markdown('<h1 class="main-header">模型评估</h1>', unsafe_allow_html=True)
        
        if len(test_data) == 0:
            st.warning("请上传测试数据集或使用示例数据。")
            st.stop()
        
        # 评估按钮
        if st.button("🚀 开始评估", key="run_evaluation"):
            with st.spinner("正在评估中..."):
                metrics, examples = run_evaluation(test_data, prompts, max_samples, seed)
                
                # 保存结果到会话状态
                st.session_state.metrics = metrics
                st.session_state.examples = examples
                st.session_state.config = {
                    "seed": seed,
                    "sample_count": min(max_samples, len(test_data)),
                    "sample_ratio": sample_ratio,
                    "device": "CPU" if not torch.cuda.is_available() else "GPU"
                }
        
        # 显示评估结果
        if 'metrics' in st.session_state and 'examples' in st.session_state:
            display_evaluation_results(st.session_state.metrics, st.session_state.examples)
            
            # 导出选项
            st.markdown("### 导出结果")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 生成JSON下载链接
                export_data = {
                    "metrics": st.session_state.metrics,
                    "config": st.session_state.config
                }
                st.markdown(
                    get_download_link(export_data, "evaluation_results.json", "📊 下载评估结果 (JSON)"),
                    unsafe_allow_html=True
                )
            
            with col2:
                # 生成HTML报告下载链接
                report_html = generate_html_report(
                    st.session_state.metrics, 
                    st.session_state.examples,
                    st.session_state.config
                )
                
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="模型评估报告.html">📄 下载完整评估报告 (HTML)</a>'
                st.markdown(href, unsafe_allow_html=True)a
    
    elif page == "数据集分析":
        st.markdown('<h1 class="main-header">数据集分析</h1>', unsafe_allow_html=True)
        
        if len(test_data) == 0:
            st.warning("请上传测试数据集或使用示例数据。")
            st.stop()
        
        display_data_analysis(test_data)
    
    # 页脚
    st.markdown("""
    <footer>
        <p>音频参数模型评估工具 | 版本 1.0 | 制作者：MixMaster团队</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()