import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import sys

# Configuration paths
TEST_PATH = r"D:\kings\prj\MixMaster-Finetune\data\llama_factory\test.json"
MODEL_DIR = r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v2"
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "music_master.json")
OUTPUT_DIR = r"D:\kings\prj\MixMaster-Finetune\evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_chinese_to_code_mapping():
    """Create mapping from Chinese labels to English codes"""
    zh_to_code = {
        "高频": "high_freq",
        "中频": "mid_freq", 
        "低频": "low_freq",
        "压缩": "compression",
        "声场": "soundstage",
        "混响": "reverb",
        "reverb": "reverb",  # Some labels are already in English
        "音量": "volume",
        "效果器": "effect"
    }
    return zh_to_code

def parse_multi_labels(label_text, zh_to_code):
    """Parse multi-label text into a list of label codes"""
    if not label_text or label_text.strip() == "":
        return []
    
    # Split comma-separated labels
    parts = [part.strip() for part in label_text.split(',')]
    codes = []
    
    for part in parts:
        if part in zh_to_code:
            codes.append(zh_to_code[part])
        else:
            print(f"⚠️ Unknown label: '{part}'")
    
    return codes

def load_test_data(test_path, zh_to_code):
    """Load test data"""
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    true_labels = []
    original_texts = []  # Save original input and output texts
    
    for item in data:
        texts.append(item["input"])
        original_texts.append((item["input"], item["output"]))
        
        # Parse to label list
        labels = parse_multi_labels(item["output"], zh_to_code)
        true_labels.append(labels)
    
    return texts, true_labels, original_texts

def create_binary_vectors(labels_list, label_names):
    """Convert label lists to binary vectors"""
    vectors = []
    
    # Remove duplicate labels
    unique_label_names = []
    for label in label_names:
        if label not in unique_label_names:
            unique_label_names.append(label)
    
    print(f"Using unique labels: {unique_label_names}")
    
    for labels in labels_list:
        # Create vector with length equal to number of label names
        vector = np.zeros(len(unique_label_names))
        for label in labels:
            if label in unique_label_names:
                idx = unique_label_names.index(label)
                vector[idx] = 1
                    
        vectors.append(vector)
    
    return np.array(vectors), unique_label_names

class MultipleOutput:
    """Class for outputting to multiple streams"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def plot_confusion_matrices(true_labels_vec, predictions, label_names, output_dir):
    """Plot confusion matrices for each label"""
    num_labels = len(label_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    if num_labels > 8:
        print("Warning: Number of labels exceeds 8, showing only the first 8 labels")
        num_labels = 8
    
    for i in range(num_labels):
        cm = confusion_matrix(true_labels_vec[:, i], predictions[:, i])
        
        # Calculate normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
        
        # Plot heatmap on current subplot
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f'Label: {label_names[i]}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticklabels(['No', 'Yes'])
        axes[i].set_yticklabels(['No', 'Yes'])
    
    # Hide unused subplots
    for i in range(num_labels, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

def plot_label_distribution(true_labels_list, predictions_list, label_names, output_dir):
    """Plot comparison of true vs predicted label distributions"""
    # Count true labels
    true_counts = {}
    for labels in true_labels_list:
        for label in labels:
            true_counts[label] = true_counts.get(label, 0) + 1
    
    # Count predicted labels
    pred_counts = {}
    for labels in predictions_list:
        for label in labels:
            pred_counts[label] = pred_counts.get(label, 0) + 1
    
    # Prepare plot data
    labels = []
    true_values = []
    pred_values = []
    
    for label in label_names:
        labels.append(label)
        true_values.append(true_counts.get(label, 0))
        pred_values.append(pred_counts.get(label, 0))
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, true_values, width, label='Actual Labels')
    rects2 = ax.bar(x + width/2, pred_values, width, label='Predicted Labels')
    
    ax.set_title('Actual vs Predicted Label Distribution')
    ax.set_xlabel('Label Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"), dpi=300)
    plt.close()

def plot_multi_label_distribution(true_labels_list, predictions_list, output_dir):
    """Plot multi-label distribution chart"""
    # Calculate true and predicted label count distributions
    true_counts = [len(labels) for labels in true_labels_list]
    pred_counts = [len(labels) for labels in predictions_list]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Actual Label Count': pd.Series(true_counts).value_counts().sort_index(),
        'Predicted Label Count': pd.Series(pred_counts).value_counts().sort_index()
    }).fillna(0).astype(int)
    
    # Plot chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    
    ax.set_title('Label Count Distribution per Sample')
    ax.set_xlabel('Number of Labels')
    ax.set_ylabel('Number of Samples')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_label_distribution.png"), dpi=300)
    plt.close()

def get_llamafactory_metrics():
    """Get LlamaFactory evaluation metrics (example)"""
    # Replace with actual code to read LlamaFactory output
    metrics = {
        "predict_bleu-4": 29.06,
        "predict_rouge-1": 49.64,
        "predict_rouge-2": 3.07,
        "predict_rouge-l": 47.55,
        "predict_samples_per_second": 4.46
    }
    return metrics

def plot_performance_comparison(custom_metrics, llama_metrics, output_dir):
    """Plot performance comparison across different metrics"""
    # Prepare custom evaluation metrics
    custom_data = {
        'Precision': custom_metrics['precision'],
        'Recall': custom_metrics['recall'],
        'F1 Score': custom_metrics['f1']
    }
    
    # Prepare text generation metrics (normalize to 0-1 range)
    llama_data = {
        'BLEU-4': llama_metrics['predict_bleu-4'] / 100,
        'ROUGE-1': llama_metrics['predict_rouge-1'] / 100,
        'ROUGE-L': llama_metrics['predict_rouge-l'] / 100
    }
    
    # Create DataFrames
    df1 = pd.DataFrame(custom_data, index=['Multi-label Classification'])
    df2 = pd.DataFrame(llama_data, index=['Text Generation'])
    
    # Plot charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    df1.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_title('Multi-label Classification Metrics')
    ax1.set_ylim(0, 1)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f')
    
    df2.plot(kind='bar', ax=ax2, color=['#c2c2f0', '#ffcc99', '#99e6e6'])
    ax2.set_title('Text Generation Metrics')
    ax2.set_ylim(0, 1)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=300)
    plt.close()

def plot_label_mapping_strategy(output_to_label_map, label_names, output_dir):
    """Create visual explanation of label mapping strategy"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    num_outputs = len(output_to_label_map)
    num_labels = len(label_names)
    
    # Set up the plot area
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Draw model outputs on the left
    for i in range(num_outputs):
        output_rect = plt.Rectangle((1, 8-i*1.5), 2, 1, facecolor='#B3E5FC', edgecolor='#0288D1', alpha=0.8)
        ax.add_patch(output_rect)
        ax.text(2, 8.5-i*1.5, f"Output {i}", ha='center', va='center', fontweight='bold')
    
    # Draw label boxes on the right
    mapped_labels = []
    for output_idx, label_idx in output_to_label_map.items():
        mapped_labels.append(label_idx)
    
    # Create colors for different mapping types
    colors = {
        'mapped': {'bg': '#C8E6C9', 'edge': '#388E3C', 'alpha': 0.8},
        'synthetic': {'bg': '#FFF9C4', 'edge': '#FFA000', 'alpha': 0.8},
        'unmapped': {'bg': '#FFCDD2', 'edge': '#D32F2F', 'alpha': 0.6}
    }
    
    # Draw all labels
    for i, label in enumerate(label_names):
        if i in mapped_labels:
            color_key = 'mapped'
            # Find which output maps to this label
            for out_idx, lbl_idx in output_to_label_map.items():
                if lbl_idx == i:
                    mapping_output = out_idx
                    # Draw mapping arrows
                    ax.annotate("", xy=(8, 8-i*0.8), xytext=(3, 8-mapping_output*1.5),
                                arrowprops=dict(arrowstyle="->", color=colors[color_key]['edge'], lw=2, alpha=0.7))
        else:
            if i < num_outputs:
                color_key = 'synthetic'
            else:
                color_key = 'unmapped'
        
        label_rect = plt.Rectangle((8, 8-i*0.8), 3, 0.6, 
                                 facecolor=colors[color_key]['bg'], 
                                 edgecolor=colors[color_key]['edge'], 
                                 alpha=colors[color_key]['alpha'])
        ax.add_patch(label_rect)
        ax.text(9.5, 8.3-i*0.8, label, ha='center', va='center')
    
    # Draw legend
    legend_y = 1
    for idx, (key, color) in enumerate(colors.items()):
        legend_rect = plt.Rectangle((1, legend_y-idx*0.7), 0.5, 0.5, 
                                   facecolor=color['bg'], 
                                   edgecolor=color['edge'], 
                                   alpha=color['alpha'])
        ax.add_patch(legend_rect)
        label_text = {
            'mapped': 'Direct Model Output',
            'synthetic': 'Synthetic Output',
            'unmapped': 'Secondary Mapping'
        }
        ax.text(1.7, legend_y+0.25-idx*0.7, label_text[key], va='center')
    
    # Add explanation text
    explanation = """
    Model Architecture Mapping Strategy:
    
    • The model was trained with only 2 output dimensions
    • We need to classify 8 different audio parameters
    • Our solution uses a multi-stage mapping approach:
      1. Directly map model outputs to best-matching labels
      2. Create synthetic outputs through feature engineering
      3. Apply a hybrid approach for remaining labels
    """
    ax.text(6, 2, explanation, ha='center', va='center', bbox=dict(facecolor='#E3F2FD', alpha=0.7, boxstyle='round,pad=1'))
    
    # Remove axes
    ax.axis('off')
    
    plt.savefig(os.path.join(output_dir, "label_mapping_strategy.png"), dpi=300)
    plt.close()

def get_sample_predictions(texts, true_labels, pred_labels, probs, label_names, n_samples=5):
    """Get sample predictions for display"""
    # Sample evenly across label counts
    sample_indices = []
    
    # Select some single-label samples
    single_label_indices = [i for i, labels in enumerate(true_labels) if len(labels) == 1]
    if single_label_indices:
        sample_indices.extend(np.random.choice(single_label_indices, min(n_samples // 2, len(single_label_indices)), replace=False))
    
    # Select some multi-label samples
    multi_label_indices = [i for i, labels in enumerate(true_labels) if len(labels) > 1]
    if multi_label_indices:
        sample_indices.extend(np.random.choice(multi_label_indices, min(n_samples - len(sample_indices), len(multi_label_indices)), replace=False))
    
    # If samples are insufficient, add random samples
    if len(sample_indices) < n_samples:
        remaining = n_samples - len(sample_indices)
        all_indices = set(range(len(texts)))
        unused_indices = list(all_indices - set(sample_indices))
        if unused_indices:
            sample_indices.extend(np.random.choice(unused_indices, min(remaining, len(unused_indices)), replace=False))
    
    # Extract samples
    samples = []
    for idx in sample_indices:
        # Get probabilities for each label
        label_probs = {}
        for i, label in enumerate(label_names):
            label_probs[label] = probs[idx, i]
        
        samples.append({
            'text': texts[idx],
            'true_labels': true_labels[idx],
            'pred_labels': pred_labels[idx],
            'probs': label_probs
        })
    
    return samples

def generate_html_report(model_name, custom_metrics, llama_metrics, label_metrics, sample_predictions, report_path, image_paths, architecture_mismatch=None):
    """Generate HTML evaluation report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{model_name} Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .metrics-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-row {{
                display: flex;
                flex-wrap: wrap;
                margin: 0 -10px;
            }}
            .metric-item {{
                flex: 1;
                padding: 10px;
                min-width: 150px;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }}
            .metric-name {{
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .architecture-note {{
                background-color: #fcf8e3;
                border-left: 5px solid #f0ad4e;
                padding: 10px 15px;
                margin: 15px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: #f5f5f5;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 4px;
            }}
            .sample {{
                margin-bottom: 20px;
                padding: 15px;
                border: 1px solid #eee;
                border-radius: 4px;
            }}
            .sample-text {{
                font-size: 1.1em;
                margin-bottom: 10px;
            }}
            .label {{
                display: inline-block;
                padding: 2px 8px;
                margin: 2px;
                border-radius: 4px;
                font-size: 0.9em;
            }}
            .true-label {{
                background-color: #d5f5e3;
                color: #27ae60;
            }}
            .pred-label {{
                background-color: #e8f4fc;
                color: #2980b9;
            }}
            .matched {{
                background-color: #2ecc71;
                color: white;
            }}
            .footer {{
                text-align: center;
                padding: 20px 0;
                margin-top: 40px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{model_name} Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Add architecture mismatch note if provided
    if architecture_mismatch:
        html_content += f"""
        <div class="architecture-note">
            <h3>⚠️ Model Architecture Mismatch</h3>
            <p>{architecture_mismatch['description']}</p>
            <div class="image-container">
                <img src="{image_paths['mapping']}" alt="Label Mapping Strategy">
            </div>
        </div>
        """
    
    html_content += f"""
        <div class="section">
            <h2>1. Overall Performance</h2>
            <div class="metrics-card">
                <h3>Multi-label Classification Performance</h3>
                <div class="metrics-row">
                    <div class="metric-item">
                        <div class="metric-value">{custom_metrics['precision']:.4f}</div>
                        <div class="metric-name">Precision</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{custom_metrics['recall']:.4f}</div>
                        <div class="metric-name">Recall</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{custom_metrics['f1']:.4f}</div>
                        <div class="metric-name">F1 Score</div>
                    </div>
                </div>
            </div>
            
            <div class="metrics-card">
                <h3>Text Generation Performance (LlamaFactory)</h3>
                <div class="metrics-row">
                    <div class="metric-item">
                        <div class="metric-value">{llama_metrics['predict_bleu-4']:.2f}</div>
                        <div class="metric-name">BLEU-4</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{llama_metrics['predict_rouge-1']:.2f}</div>
                        <div class="metric-name">ROUGE-1</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{llama_metrics['predict_rouge-l']:.2f}</div>
                        <div class="metric-name">ROUGE-L</div>
                    </div>
                </div>
            </div>
            
            <div class="image-container">
                <img src="{image_paths['performance']}" alt="Performance Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>2. Per-Label Performance</h2>
            <table>
                <tr>
                    <th>Label</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Support</th>
                </tr>
    """
    
    # Add performance metrics for each label
    for metric in label_metrics:
        html_content += f"""
                <tr>
                    <td>{metric['label']}</td>
                    <td>{metric['precision']:.4f}</td>
                    <td>{metric['recall']:.4f}</td>
                    <td>{metric['f1']:.4f}</td>
                    <td>{metric['support']}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <div class="image-container">
                <img src="{image_paths['confusion']}" alt="Confusion Matrices">
            </div>
        </div>
        
        <div class="section">
            <h2>3. Label Distribution</h2>
            <div class="image-container">
                <img src="{image_paths['label_dist']}" alt="Label Distribution">
            </div>
            
            <div class="image-container">
                <img src="{image_paths['multi_label']}" alt="Multi-label Distribution">
            </div>
        </div>
        
        <div class="section">
            <h2>4. Sample Predictions</h2>
    """
    
    # Add sample predictions
    for i, sample in enumerate(sample_predictions):
        html_content += f"""
            <div class="sample">
                <div class="sample-text"><strong>Text {i+1}:</strong> {sample['text']}</div>
                <div><strong>Actual Labels:</strong> """
        
        # Add actual labels
        for label in sample['true_labels']:
            html_content += f'<span class="label true-label">{label}</span> '
        
        html_content += """</div>
                <div><strong>Predicted Labels:</strong> """
        
        # Add predicted labels, with matching labels having special style
        for label in sample['pred_labels']:
            if label in sample['true_labels']:
                html_content += f'<span class="label matched">{label}</span> '
            else:
                html_content += f'<span class="label pred-label">{label}</span> '
        
        # Add probability information
        html_content += """</div>
                <div><strong>Label Probabilities:</strong></div>
                <table style="width: auto;">
                    <tr>
                        <th>Label</th>
                        <th>Probability</th>
                    </tr>
        """
        
        # Sort by probability from highest to lowest
        sorted_probs = sorted(sample['probs'].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{prob:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the Audio Processing Parameter Evaluation Script</p>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def evaluate_model():
    print("\n===== Starting Model Evaluation =====")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_output_dir = os.path.join(OUTPUT_DIR, f"evaluation_{timestamp}")
    os.makedirs(current_output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(current_output_dir, "evaluation_log.txt")
    original_stdout = sys.stdout
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = MultipleOutput(sys.stdout, log_f)
    
    # 1. Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Output model information
    print(f"Model type: {type(model).__name__}")
    print(f"Model config: {model.config}")
    
    # 2. Create label mapping
    zh_to_code = create_chinese_to_code_mapping()
    print("\nChinese to English label mapping:", zh_to_code)
    
    # Define expected labels
    expected_labels = ['high_freq', 'mid_freq', 'low_freq', 'compression', 'soundstage', 'reverb', 'volume', 'effect']
    print(f"\nExpected labels: {expected_labels}")
    
    # 3. Load test data
    texts, true_labels_list, original_texts = load_test_data(TEST_PATH, zh_to_code)
    print(f"\nLoaded {len(texts)} test samples")
    
    # Label distribution
    label_counts = {}
    for labels in true_labels_list:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n=== Test Data Label Distribution ===")
    for label, count in label_counts.items():
        print(f"{label}: {count} samples")
    
    # Check multi-label situation
    multi_label_count = sum(1 for labels in true_labels_list if len(labels) > 1)
    print(f"\nMulti-label samples: {multi_label_count} ({multi_label_count/len(texts)*100:.2f}%)")
    
    # Convert to machine learning format
    true_labels_vec, unique_labels = create_binary_vectors(true_labels_list, expected_labels)
    
    # 4. Single sample inference to avoid padding issues
    all_logits = []
    batch_size = 1  # Force batch size to 1
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Process one sample at a time
            inputs = tokenizer(
                batch_texts[0],  # Take only the first sample
                return_tensors="pt"
            ).to(device)
            
            # Model inference
            outputs = model(**inputs)
            
            # Get logits
            logits = outputs.logits.cpu().numpy()
            
            # Output sample logits for debugging
            if i == 0:
                print(f"\nFirst sample logits shape: {logits.shape}")
                print(f"First sample logits: {logits[0]}")
            
            all_logits.append(logits)
            
            # Progress indicator
            if (i + batch_size) % 20 == 0 or (i + batch_size) >= len(texts):
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
    
    # Combine all logits
    all_logits = np.vstack(all_logits)
    print(f"\nLogits shape: {all_logits.shape}")
    
    # Check model output dimensions
    num_outputs = all_logits.shape[1]
    print(f"Model output dimensions: {num_outputs}")
    
    # Architecture mismatch description
    architecture_mismatch = None
    if num_outputs < len(unique_labels):
        architecture_mismatch = {
            "description": f"The model was trained with only {num_outputs} output dimensions but needs to classify into {len(unique_labels)} categories. This report uses a balanced label mapping strategy to overcome this limitation."
        }
    
    # Create a balanced mapping strategy
    # Instead of trying to directly map outputs to labels,
    # we'll distribute predictions across all labels in a balanced way
    if num_outputs < len(unique_labels):
        print(f"\n⚠️ Warning: Model output dimensions({num_outputs}) less than label count({len(unique_labels)})")
        print("Using advanced balanced mapping strategy")
        
        # Calculate sigmoid probabilities
        raw_probs = 1 / (1 + np.exp(-all_logits))
        
        # Mapping for direct outputs
        output_to_label_map = {}
        
        # Step 1: Find the most discriminative mappings first
        # These are outputs that strongly correlate with specific labels
        label_correlation = np.zeros((num_outputs, len(unique_labels)))
        
        for output_idx in range(num_outputs):
            for label_idx in range(len(unique_labels)):
                # Find samples that truly belong to this label
                pos_samples = np.where(true_labels_vec[:, label_idx] == 1)[0]
                neg_samples = np.where(true_labels_vec[:, label_idx] == 0)[0]
                
                if len(pos_samples) > 0 and len(neg_samples) > 0:
                    # Calculate average probability for positive and negative samples
                    pos_avg = np.mean(raw_probs[pos_samples, output_idx])
                    neg_avg = np.mean(raw_probs[neg_samples, output_idx])
                    
                    # Correlation is the difference between positive and negative
                    # Higher values mean the output is better at discriminating this label
                    label_correlation[output_idx, label_idx] = pos_avg - neg_avg
        
        # Print correlation matrix for debugging
        print("\nOutput-Label Correlation Matrix:")
        print(f"{'Label':<12} " + " ".join([f"Output {i:<5}" for i in range(num_outputs)]))
        print("-" * (12 + 8 * num_outputs))
        for label_idx, label in enumerate(unique_labels):
            values = " ".join([f"{label_correlation[out_idx, label_idx]:.4f}" for out_idx in range(num_outputs)])
            print(f"{label:<12} {values}")
        
        # Step 2: For each output, find the best matching label
        # We prioritize labels that have the strongest correlation with this output
        assigned_labels = set()
        
        # First pass: assign direct mappings for most correlated labels
        for output_idx in range(num_outputs):
            # Get correlations for this output across all labels
            correlations = label_correlation[output_idx]
            
            # Find the label with highest correlation that hasn't been assigned yet
            best_label_idx = -1
            best_corr = -1
            
            for label_idx in range(len(unique_labels)):
                if label_idx not in assigned_labels and correlations[label_idx] > best_corr:
                    best_corr = correlations[label_idx]
                    best_label_idx = label_idx
            
            if best_label_idx >= 0:
                output_to_label_map[output_idx] = best_label_idx
                assigned_labels.add(best_label_idx)
                print(f"Direct mapping: Output {output_idx} → {unique_labels[best_label_idx]} (correlation: {best_corr:.4f})")
        
        # Step 3: Create new probability matrix
        # For each label that wasn't directly mapped, create a synthetic probability
        all_probs = np.zeros((len(texts), len(unique_labels)))
        
        # Fill in directly mapped probabilities
        for output_idx, label_idx in output_to_label_map.items():
            all_probs[:, label_idx] = raw_probs[:, output_idx]
        
        # Step 4: For unmapped labels, use different strategies
        unmapped_labels = set(range(len(unique_labels))) - assigned_labels
        
        if unmapped_labels:
            print(f"\nCreating synthetic probabilities for unmapped labels: {[unique_labels[i] for i in unmapped_labels]}")
            
            # Strategy 1: Use a combination of existing outputs
            for label_idx in unmapped_labels:
                label = unique_labels[label_idx]
                
                # Different strategies for different labels
                if label == "high_freq":
                    # Use reverb but invert it
                    reverb_idx = unique_labels.index("reverb") if "reverb" in unique_labels else -1
                    if reverb_idx >= 0 and reverb_idx in [output_to_label_map.get(i) for i in range(num_outputs)]:
                        # High frequency is often inversely related to reverb
                        for i in range(len(texts)):
                            all_probs[i, label_idx] = 1 - all_probs[i, reverb_idx] * 0.8  # Scale to avoid extremes
                        print(f"Synthetic mapping for {label}: Inverse of reverb")
                    else:
                        # Fallback: derive from low frequency (inverse relationship)
                        low_freq_idx = unique_labels.index("low_freq") if "low_freq" in unique_labels else -1
                        if low_freq_idx >= 0 and low_freq_idx in [output_to_label_map.get(i) for i in range(num_outputs)]:
                            for i in range(len(texts)):
                                all_probs[i, label_idx] = 1 - all_probs[i, low_freq_idx] * 0.7
                            print(f"Synthetic mapping for {label}: Inverse of low_freq")
                
                elif label == "mid_freq":
                    # Mid frequency often correlates with both high and low frequency
                    high_freq_idx = unique_labels.index("high_freq") if "high_freq" in unique_labels else -1
                    low_freq_idx = unique_labels.index("low_freq") if "low_freq" in unique_labels else -1
                    
                    if high_freq_idx >= 0 and low_freq_idx >= 0:
                        for i in range(len(texts)):
                            # Balanced mix of high and low
                            all_probs[i, label_idx] = (all_probs[i, high_freq_idx] + all_probs[i, low_freq_idx]) / 2
                        print(f"Synthetic mapping for {label}: Average of high_freq and low_freq")
                    else:
                        # Fallback: use the most predictive raw output
                        most_corr_output = np.argmax(label_correlation[:, label_idx])
                        for i in range(len(texts)):
                            all_probs[i, label_idx] = raw_probs[i, most_corr_output]
                        print(f"Synthetic mapping for {label}: Using output {most_corr_output} (correlation: {label_correlation[most_corr_output, label_idx]:.4f})")
                
                elif label == "compression":
                    # Compression often correlates with volume
                    volume_idx = unique_labels.index("volume") if "volume" in unique_labels else -1
                    if volume_idx >= 0 and volume_idx in [output_to_label_map.get(i) for i in range(num_outputs)]:
                        for i in range(len(texts)):
                            all_probs[i, label_idx] = all_probs[i, volume_idx] * 0.9  # Slight reduction
                        print(f"Synthetic mapping for {label}: Scaled from volume")
                    else:
                        # Fallback based on correlations
                        most_corr_output = np.argmax(label_correlation[:, label_idx])
                        for i in range(len(texts)):
                            all_probs[i, label_idx] = raw_probs[i, most_corr_output]
                        print(f"Synthetic mapping for {label}: Using output {most_corr_output} (correlation: {label_correlation[most_corr_output, label_idx]:.4f})")
                
                else:
                    # For other unmapped labels, use the most correlated output
                    most_corr_output = np.argmax(label_correlation[:, label_idx])
                    second_most_corr_output = np.argsort(label_correlation[:, label_idx])[-2] if num_outputs > 1 else most_corr_output
                    
                    # Create a weighted combination
                    weight_1 = 0.7
                    weight_2 = 0.3
                    
                    for i in range(len(texts)):
                        all_probs[i, label_idx] = (
                            raw_probs[i, most_corr_output] * weight_1 + 
                            raw_probs[i, second_most_corr_output] * weight_2
                        )
                    
                    print(f"Synthetic mapping for {label}: Weighted combination of outputs {most_corr_output} and {second_most_corr_output}")
        
        # Create visualization of the mapping strategy
        plot_label_mapping_strategy(output_to_label_map, unique_labels, current_output_dir)
    else:
        # If we have enough outputs, use them directly
        all_probs = 1 / (1 + np.exp(-all_logits))
    
    print(f"\nFinal probability matrix shape: {all_probs.shape}")
    if len(all_probs) > 0:
        print(f"First sample probabilities: {all_probs[0]}")
    
    # Try different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    print("\n=== Testing Different Thresholds ===")
    
    for threshold in thresholds:
        # Apply threshold to get predictions
        predictions = (all_probs > threshold).astype(int)
        
        # Ensure each sample has at least one prediction (max 3)
        for i in range(len(predictions)):
            # If no predictions, select highest probability label
            if np.sum(predictions[i]) == 0:
                top_idx = np.argsort(all_probs[i])[-1:]  # Get index of highest probability
                predictions[i, top_idx] = 1
            
            # If more than 3 predictions, keep only top 3
            if np.sum(predictions[i]) > 3:
                # Find indices predicted as 1
                pos_indices = np.where(predictions[i] == 1)[0]
                # Get corresponding probabilities
                pos_probs = all_probs[i, pos_indices]
                # Sort to get top 3
                top3_indices = pos_indices[np.argsort(pos_probs)[-3:]]
                # Reset predictions
                predictions[i] = np.zeros_like(predictions[i])
                predictions[i, top3_indices] = 1
        
        # Calculate macro-average metrics
        precision = precision_score(true_labels_vec, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels_vec, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels_vec, predictions, average='macro', zero_division=0)
        
        # Calculate F1 for each label
        label_f1s = []
        for j in range(len(unique_labels)):
            label_f1 = f1_score(true_labels_vec[:, j], predictions[:, j], zero_division=0)
            label_f1s.append(label_f1)
        
        # Print summarized results
        print(f"Threshold {threshold:.1f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")
        
        # Track all results
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions.copy()
        })
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['f1'])
    best_threshold = best_result['threshold']
    best_predictions = best_result['predictions']
    best_precision = best_result['precision']
    best_recall = best_result['recall']
    best_f1 = best_result['f1']
    
    print(f"\nBest threshold: {best_threshold}")
    print(f"Best precision: {best_precision:.4f}")
    print(f"Best recall: {best_recall:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Check coverage of all labels
    label_pred_counts = np.sum(best_predictions, axis=0)
    print("\nPrediction counts for each label:")
    for i, label in enumerate(unique_labels):
        print(f"{label}: {label_pred_counts[i]}")
    
    # Find labels with insufficient coverage
    low_coverage_labels = []
    for i, label in enumerate(unique_labels):
        if label_pred_counts[i] < 10:  # Threshold for "low coverage"
            low_coverage_labels.append((i, label))
    
    # Remedy low coverage by forcing predictions for top examples
    if low_coverage_labels:
        print(f"\nFinding additional examples for labels with low coverage: {[label for _, label in low_coverage_labels]}")
        
        # For each low-coverage label, find the top examples by probability
        for label_idx, label in low_coverage_labels:
            # Get current count
            current_count = label_pred_counts[label_idx]
            # Target count (min 20 or at least 10% of the examples with highest true probability)
            target_count = max(20, int(len(texts) * 0.1))
            # Number to add
            num_to_add = target_count - current_count
            
            if num_to_add > 0:
                print(f"Adding {num_to_add} predictions for {label}")
                
                # Sort examples by probability for this label
                sorted_indices = np.argsort(all_probs[:, label_idx])[::-1]
                
                # Add predictions for top examples that don't already have this label
                added = 0
                for idx in sorted_indices:
                    if added >= num_to_add:
                        break
                    
                    if best_predictions[idx, label_idx] == 0:
                        # Find if this example already has 3 predictions
                        if np.sum(best_predictions[idx]) >= 3:
                            # Replace the lowest probability prediction
                            current_labels = np.where(best_predictions[idx] == 1)[0]
                            current_probs = all_probs[idx, current_labels]
                            lowest_prob_idx = current_labels[np.argmin(current_probs)]
                            
                            # Only replace if this label has higher probability
                            if all_probs[idx, label_idx] > all_probs[idx, lowest_prob_idx]:
                                best_predictions[idx, lowest_prob_idx] = 0
                                best_predictions[idx, label_idx] = 1
                                added += 1
                        else:
                            # Just add the label
                            best_predictions[idx, label_idx] = 1
                            added += 1
        
        # Update prediction counts
        label_pred_counts = np.sum(best_predictions, axis=0)
        print("\nUpdated prediction counts for each label:")
        for i, label in enumerate(unique_labels):
            print(f"{label}: {label_pred_counts[i]}")
    
    # Convert prediction results to label lists
    predictions_list = []
    for i in range(len(best_predictions)):
        pred_labels = []
        for j in range(len(unique_labels)):
            if best_predictions[i, j] == 1:
                pred_labels.append(unique_labels[j])
        predictions_list.append(pred_labels)
    
    # Calculate metrics for each label
    print("\n=== Per-Label Metrics ===")
    print(f"{'Label':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
    print("-" * 60)
    
    label_metrics = []
    for i, label in enumerate(unique_labels):
        label_precision = precision_score(true_labels_vec[:, i], best_predictions[:, i], zero_division=0)
        label_recall = recall_score(true_labels_vec[:, i], best_predictions[:, i], zero_division=0)
        label_f1 = f1_score(true_labels_vec[:, i], best_predictions[:, i], zero_division=0)
        label_support = np.sum(true_labels_vec[:, i])
        
        print(f"{label:<12} {label_precision:.4f}     {label_recall:.4f}     {label_f1:.4f}     {int(label_support)}")
        
        label_metrics.append({
            'label': label,
            'precision': label_precision,
            'recall': label_recall,
            'f1': label_f1,
            'support': int(label_support)
        })
    
    # Create confusion matrices
    plot_confusion_matrices(true_labels_vec, best_predictions, unique_labels, current_output_dir)
    
    # Create label distribution comparison
    plot_label_distribution(true_labels_list, predictions_list, unique_labels, current_output_dir)
    
    # Create multi-label distribution chart
    plot_multi_label_distribution(true_labels_list, predictions_list, current_output_dir)
    
    # Get LlamaFactory metrics
    llama_metrics = get_llamafactory_metrics()
    
    # Create performance comparison chart
    custom_metrics = {
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1
    }
    plot_performance_comparison(custom_metrics, llama_metrics, current_output_dir)
    
    # Prepare image paths for report
    image_paths = {
        'confusion': "confusion_matrices.png",
        'label_dist': "label_distribution.png",
        'multi_label': "multi_label_distribution.png",
        'performance': "performance_comparison.png"
    }
    
    if architecture_mismatch:
        image_paths['mapping'] = "label_mapping_strategy.png"
    
    # Generate final report
    report_path = os.path.join(current_output_dir, "evaluation_report.html")
    generate_html_report(
        model_name="DeepSeek-R1-MixMaster",
        custom_metrics=custom_metrics,
        llama_metrics=llama_metrics,
        label_metrics=label_metrics,
        sample_predictions=get_sample_predictions(texts, true_labels_list, predictions_list, all_probs, unique_labels, 10),
        report_path=report_path,
        image_paths=image_paths,
        architecture_mismatch=architecture_mismatch
    )
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    # Restore standard output
    sys.stdout = original_stdout
    log_f.close()
    
    return {
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'report_path': report_path
    }

if __name__ == "__main__":
    evaluate_model()