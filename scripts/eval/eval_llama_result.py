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
    'font.family': 'serif',  # LaTeX-compatible font
    'font.serif': ['Computer Modern Roman', 'Times', 'Palatino', 'DejaVu Serif'],
    'text.usetex': False,     # Set to True if you have LaTeX installed
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
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
    'predict_model_preparation_time': 'Model Prep Time (s)',
    'predict_runtime': 'Runtime (s)',
    'predict_samples_per_second': 'Samples/second',
    'predict_steps_per_second': 'Steps/second'
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

# Define a beautiful color palette suitable for academic publications
colors = ['#4e79a7', '#f28e2c', '#76b7b2']  # Blue, Orange, Teal - colorblind friendly

# Create two separate plots with improved aesthetics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
fig.suptitle('Model Evaluation Metrics Comparison', fontweight='bold', y=0.98)

# Plot 1: Performance metrics (BLEU, ROUGE)
x1 = np.arange(len(performance_metrics))
width = 0.2
multiplier = 0

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in performance_metrics]
    bars = ax1.bar(x1 + offset, values, width, label=model_name, color=colors[i], 
            edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    multiplier += 1

ax1.set_xlabel('Metrics', fontweight='bold')
ax1.set_ylabel('Scores', fontweight='bold')
ax1.set_title('Performance Metrics (higher is better)', fontsize=11)
ax1.set_xticks(x1 + width)
ax1.set_xticklabels([metric_display_names[m] for m in performance_metrics])
ax1.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Speed metrics with diverging colors
x2 = np.arange(len(speed_metrics))
multiplier = 0

for i, (model_name, result) in enumerate(results.items()):
    offset = width * multiplier
    values = [result[metric] for metric in speed_metrics]
    bars = ax2.bar(x2 + offset, values, width, label=model_name, color=colors[i],
            edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    multiplier += 1

ax2.set_xlabel('Metrics', fontweight='bold')
ax2.set_ylabel('Values', fontweight='bold')
ax2.set_title('Speed and Time Metrics', fontsize=11)
ax2.set_xticks(x2 + width)
ax2.set_xticklabels([metric_display_names[m] for m in speed_metrics])
ax2.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray')
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_comparison_all_metrics.pdf', bbox_inches='tight')
plt.savefig('model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

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

# Create a radar chart for even more impressive visualization
# First, reformat the data for radar chart
metrics_for_radar = ['BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 
                     'Samples/second', 'Steps/second']

# Create a copy with only the metrics we want for the radar chart
radar_df = df.loc[metrics_for_radar].copy()

# For time metrics, higher should be better, so invert them
for idx in radar_df.index:
    if "Time" in idx or "Runtime" in idx:
        radar_df.loc[idx] = 1 / radar_df.loc[idx]

# Normalize all values to 0-1 range for radar chart
for idx in radar_df.index:
    max_val = radar_df.loc[idx].max()
    if max_val > 0:
        radar_df.loc[idx] = radar_df.loc[idx] / max_val

# Set data
categories = metrics_for_radar
N = len(categories)

# Create angles for each metric
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Add background grid
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=11)
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], size=8)
plt.ylim(0, 1)

# Plot each model
for i, model_name in enumerate(radar_df.columns):
    values = radar_df[model_name].values.tolist()
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(0.15, 0.95), frameon=True)
plt.title('Model Performance Comparison (Normalized)', size=15, fontweight='bold', y=1.08)

# Add grid lines and styling
ax.grid(True, linestyle='--', alpha=0.7)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('model_radar_comparison.pdf', bbox_inches='tight')
plt.savefig('model_radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()