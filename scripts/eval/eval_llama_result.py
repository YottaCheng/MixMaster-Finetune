import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Create two separate plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Performance metrics (BLEU, ROUGE)
x1 = np.arange(len(performance_metrics))
width = 0.2
multiplier = 0

for model_name, result in results.items():
    offset = width * multiplier
    values = [result[metric] for metric in performance_metrics]
    ax1.bar(x1 + offset, values, width, label=model_name)
    multiplier += 1

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Scores')
ax1.set_title('Model Evaluation Performance Metrics Comparison')
ax1.set_xticks(x1 + width, [metric_display_names[m] for m in performance_metrics])
ax1.legend(loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Speed metrics
x2 = np.arange(len(speed_metrics))
multiplier = 0

for model_name, result in results.items():
    offset = width * multiplier
    values = [result[metric] for metric in speed_metrics]
    ax2.bar(x2 + offset, values, width, label=model_name)
    multiplier += 1

ax2.set_xlabel('Metrics')
ax2.set_ylabel('Values')
ax2.set_title('Model Evaluation Speed and Time Metrics Comparison')
ax2.set_xticks(x2 + width, [metric_display_names[m] for m in speed_metrics])
ax2.legend(loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create a heatmap for better visualization of relative performance
plt.figure(figsize=(12, 8))
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

# Convert DataFrame to numpy array for imshow
heatmap_data = normalized_df.to_numpy()
heatmap = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(heatmap, label='Normalized Score (higher is better)')

# Add labels
plt.xticks(np.arange(len(normalized_df.columns)), normalized_df.columns, rotation=45)
plt.yticks(np.arange(len(normalized_df.index)), normalized_df.index)

# Add text annotations on the heatmap
for i in range(len(normalized_df.index)):
    for j in range(len(normalized_df.columns)):
        original_value = float(df.iloc[i, j])
        plt.text(j, i, f"{original_value:.2f}", ha="center", va="center", color="white")

plt.title('Model Comparison Heatmap (Normalized Scores)')
plt.tight_layout()
plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the comparison table
print("\nMetric Comparison Table:")
print(df.to_string())

# Export to Excel for easier analysis
df.to_excel('model_comparison_metrics.xlsx')
print("\nData exported to 'model_comparison_metrics.xlsx'")