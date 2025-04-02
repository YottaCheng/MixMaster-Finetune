import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set the style to a clean, publication-ready format
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300
})

def load_all_results(filepath):
    """Load the all_results.json file from the given filepath."""
    with open(os.path.join(filepath, 'all_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

# Paths to the model evaluation results
paths = {
    'Qwen2-1.5B': 'D:/kings/prj/Finetune_local/LLaMA-Factory/saves/Qwen2-1.5B/lora/eval_Qwen2',
    'DeepSeek-Origin': 'D:/kings/prj/Finetune_local/LLaMA-Factory/saves/DeepSeek-R1-1.5B-Distill/lora/eval_DeepSeek_Origin',
    'DeepSeek-FineTuning': 'D:/kings/prj/Finetune_local/LLaMA-Factory/saves/DeepSeek-R1-1.5B-Distill/lora/eval_DeepSeek_FineTuning'
}

# Load all results from each model
results = {}
for model_name, path in paths.items():
    results[model_name] = load_all_results(path)

# All metrics we want to display
metrics = [
    'predict_bleu-4',
    'predict_rouge-1', 
    'predict_rouge-2', 
    'predict_rouge-l',
    'predict_model_preparation_time',
    'predict_runtime',
    'predict_samples_per_second',
    'predict_steps_per_second'
]

# More readable metric names for display
metric_display_names = {
    'predict_bleu-4': 'BLEU-4',
    'predict_rouge-1': 'ROUGE-1',
    'predict_rouge-2': 'ROUGE-2', 
    'predict_rouge-l': 'ROUGE-L',
    'predict_model_preparation_time': 'Prep Time (s)',
    'predict_runtime': 'Runtime (s)',
    'predict_samples_per_second': 'Samples/sec',
    'predict_steps_per_second': 'Steps/sec'
}

# Create a DataFrame for easy comparison
df = pd.DataFrame(columns=list(paths.keys()), index=[metric_display_names[m] for m in metrics])

for model_name, result in results.items():
    for metric in metrics:
        # Ensure value is numeric by explicitly converting to float
        df.loc[metric_display_names[metric], model_name] = float(result[metric])

# Split metrics into two groups: performance metrics and speed metrics
performance_metrics = metrics[:4]  # BLEU and ROUGE metrics
speed_metrics = metrics[4:]        # Time and throughput metrics

# Define a softer color palette (lower saturation)
colors = ['#8cacd0', '#f7bc7f', '#a9d6ca']  # Softer Blue, Orange, Teal

# Create separate figures for each metric group to avoid overlap
# First figure: Performance metrics
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
fig1.suptitle('Performance Metrics (higher is better)', y=0.98)

# Add a light background to the plot area
ax1.set_facecolor('#f8f8f8')

x1 = np.arange(len(performance_metrics))
width = 0.25
multiplier = 0

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in performance_metrics]
    bars = ax1.bar(x1 + offset, values, width, label=model_name, color=colors[i], 
            edgecolor='dimgray', linewidth=0.5, alpha=0.9)
    
    # Add value labels with better positioning
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # Adjust the vertical position based on the value
        if value > 15:
            v_pos = height * 0.5  # Middle of the bar for tall bars
            color = 'white'
        else:
            v_pos = height + 0.5  # Above the bar for short bars
            color = 'black'
            
        ax1.text(bar.get_x() + bar.get_width()/2., v_pos,
                f'{value:.2f}', ha='center', va='center', 
                fontsize=8, color=color)
    
    multiplier += 1

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Scores')
ax1.set_xticks(x1 + width)
ax1.set_xticklabels([metric_display_names[m] for m in performance_metrics])

# Position legend at the bottom outside the plot
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, 
           framealpha=0.95, edgecolor='lightgray', ncol=3)

ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Set better y-axis limits with some padding
y_max = max([result[metric] for model_name, result in results.items() for metric in performance_metrics])
ax1.set_ylim(0, y_max * 1.2)

plt.tight_layout()
plt.savefig('performance_metrics.pdf', bbox_inches='tight')
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')

# Second figure: Speed metrics
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
fig2.suptitle('Speed and Time Metrics', y=0.98)

# Add a light background to the plot area
ax2.set_facecolor('#f8f8f8')

x2 = np.arange(len(speed_metrics))
multiplier = 0

# For better visualization, create a log scale for vastly different values
log_scale = True

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in speed_metrics]
    
    # Apply log transform if values are vastly different (optional)
    if log_scale and max(values) > 100 * min(filter(lambda x: x > 0, values)):
        ax2.set_yscale('log')
    
    bars = ax2.bar(x2 + offset, values, width, label=model_name, color=colors[i],
            edgecolor='dimgray', linewidth=0.5, alpha=0.9)
    
    # Add value labels
    for j, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if height < 0.1:  # Very small values
            v_pos = height * 2
            color = 'black'
        elif height > 1000:  # Very large values
            v_pos = height / 2
            color = 'white'
        else:  # Medium values
            v_pos = height * 1.1
            color = 'black'
        
        # Format value based on size
        if value >= 1000:
            value_text = f'{value:.0f}'
        elif value >= 1:
            value_text = f'{value:.2f}'
        else:
            value_text = f'{value:.3f}'
            
        ax2.text(bar.get_x() + bar.get_width()/2., v_pos,
                value_text, ha='center', va='center', 
                fontsize=8, color=color, rotation=0)
    
    multiplier += 1

ax2.set_xlabel('Metrics')
ax2.set_ylabel('Values')
ax2.set_xticks(x2 + width)
ax2.set_xticklabels([metric_display_names[m] for m in speed_metrics])

# Position legend at the bottom
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, 
           framealpha=0.95, edgecolor='lightgray', ncol=3)

ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust y-axis based on whether we're using log scale
if not log_scale:
    y_max = max([result[metric] for model_name, result in results.items() for metric in speed_metrics])
    ax2.set_ylim(0, y_max * 1.2)

plt.tight_layout()
plt.savefig('speed_metrics.pdf', bbox_inches='tight')
plt.savefig('speed_metrics.png', dpi=300, bbox_inches='tight')

# Combined figure for the paper
fig, (ax1_combined, ax2_combined) = plt.subplots(2, 1, figsize=(8, 9))
fig.suptitle('Model Evaluation Metrics Comparison', y=0.98)

# Copy the content from the individual plots
# Performance metrics
ax1_combined.set_facecolor('#f8f8f8')
multiplier = 0

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in performance_metrics]
    bars = ax1_combined.bar(x1 + offset, values, width, label=model_name, color=colors[i], 
            edgecolor='dimgray', linewidth=0.5, alpha=0.9)
    
    # Add value labels - positioned outside bars for clarity
    for bar, value in zip(bars, values):
        height = bar.get_height()
        v_pos = height + (y_max * 0.02)  # Always place above the bar
        ax1_combined.text(bar.get_x() + bar.get_width()/2., v_pos,
                f'{value:.2f}', ha='center', va='bottom', 
                fontsize=7, color='black', rotation=45)
    
    multiplier += 1

ax1_combined.set_xlabel('Metrics')
ax1_combined.set_ylabel('Scores')
ax1_combined.set_title('Performance Metrics (higher is better)')
ax1_combined.set_xticks(x1 + width)
ax1_combined.set_xticklabels([metric_display_names[m] for m in performance_metrics])
ax1_combined.legend(loc='upper right', frameon=True, framealpha=0.9)
ax1_combined.grid(axis='y', linestyle='--', alpha=0.3)
ax1_combined.spines['top'].set_visible(False)
ax1_combined.spines['right'].set_visible(False)
ax1_combined.set_ylim(0, y_max * 1.2)

# Speed metrics
ax2_combined.set_facecolor('#f8f8f8')
multiplier = 0

# Use log scale for the combined plot if values are vastly different
if log_scale and max([result[metric] for model_name, result in results.items() 
                     for metric in speed_metrics]) > 100 * min(filter(lambda x: x > 0, 
                     [result[metric] for model_name, result in results.items() for metric in speed_metrics])):
    ax2_combined.set_yscale('log')

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in speed_metrics]
    bars = ax2_combined.bar(x2 + offset, values, width, label=model_name, color=colors[i],
            edgecolor='dimgray', linewidth=0.5, alpha=0.9)
    
    # Add value labels with rotated text for better fit
    for j, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if log_scale:
            if value < 0.1:  # Very small values
                v_pos = value * 2
            elif value > 1000:  # Very large values
                v_pos = value / 2
            else:  # Medium values
                v_pos = value * 1.1
        else:
            v_pos = height + 0.5
        
        # Format value based on size
        if value >= 1000:
            value_text = f'{value:.0f}'
        elif value >= 1:
            value_text = f'{value:.2f}'
        else:
            value_text = f'{value:.3f}'
            
        ax2_combined.text(bar.get_x() + bar.get_width()/2., v_pos,
                value_text, ha='center', va='bottom', 
                fontsize=7, color='black', rotation=45)
    
    multiplier += 1

ax2_combined.set_xlabel('Metrics')
ax2_combined.set_ylabel('Values')
ax2_combined.set_title('Speed and Time Metrics')
ax2_combined.set_xticks(x2 + width)
ax2_combined.set_xticklabels([metric_display_names[m] for m in speed_metrics])
ax2_combined.legend(loc='upper right', frameon=True, framealpha=0.9)
ax2_combined.grid(axis='y', linestyle='--', alpha=0.3)
ax2_combined.spines['top'].set_visible(False)
ax2_combined.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # Add more space between plots
plt.savefig('model_comparison_all_metrics.pdf', bbox_inches='tight')
plt.savefig('model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Keep the existing heatmap code here...
# Create an enhanced heatmap for better visualization of relative performance
plt.figure(figsize=(9, 6))

# Normalize the data for better visualization
normalized_df = df.copy()

# First convert all values to float
for col in normalized_df.columns:
    normalized_df[col] = normalized_df[col].astype(float)

# Then normalize
for idx in normalized_df.index:
    # For time metrics, lower is better, so invert the normalization
    if "Time" in idx or "Runtime" in idx:
        max_val = normalized_df.loc[idx].max()
        normalized_df.loc[idx] = max_val / normalized_df.loc[idx]
    else:
        # For other metrics (higher is better), normalize normally
        max_val = normalized_df.loc[idx].max()
        if max_val > 0:  # Prevent division by zero
            normalized_df.loc[idx] = normalized_df.loc[idx] / max_val

# Create a custom diverging colormap suitable for publications
cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                          ['#053061', '#2166ac', '#4393c3', 
                                           '#92c5de', '#d1e5f0', '#f7f7f7',
                                           '#fddbc7', '#f4a582', '#d6604d', 
                                           '#b2182b', '#67001f'])

# Enhanced heatmap with seaborn
ax = sns.heatmap(normalized_df, annot=df.round(2), fmt='.2f', cmap=cmap,
                annot_kws={"size": 9}, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Normalized Score (higher is better)'})

plt.title('Model Comparison (Normalized Scores)', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('model_comparison_heatmap.pdf', bbox_inches='tight')
plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the comparison table
print("\nMetric Comparison Table:")
print(df.to_string())

# Export to Excel for easier analysis
df.to_excel('model_comparison_metrics.xlsx')
print("\nData exported to 'model_comparison_metrics.xlsx'")