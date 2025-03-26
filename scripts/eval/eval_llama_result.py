import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_all_results(filepath):
    """Load the all_results.json file from the given filepath."""
    with open(os.path.join(filepath, 'all_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_generated_predictions(filepath, limit=3):
    """Load the first N entries from generated_predictions.jsonl file."""
    predictions = []
    with open(os.path.join(filepath, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            predictions.append(json.loads(line))
    return predictions

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

# Extract metrics we're interested in
metrics = ['predict_bleu-4', 'predict_rouge-1', 'predict_rouge-2', 'predict_rouge-l']
metric_names = ['BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']

# Create a DataFrame for easy comparison
df = pd.DataFrame(index=metric_names)

for model_name, result in results.items():
    model_metrics = [result[metric] for metric in metrics]
    df[model_name] = model_metrics

# Plot the results
plt.figure(figsize=(12, 8))
x = np.arange(len(metric_names))
width = 0.2
multiplier = 0

for model_name, result in results.items():
    offset = width * multiplier
    model_metrics = [result[metric] for metric in metrics]
    plt.bar(x + offset, model_metrics, width, label=model_name)
    multiplier += 1

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Evaluation Results Comparison')
plt.xticks(x + width, metric_names)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the comparison table
print("\nMetric Comparison Table:")
print(df.to_string())

# Load and examine first 3 examples from one of the models
print("\nExample Predictions (First 3) from DeepSeek-FineTuning:")
predictions = load_generated_predictions(paths['DeepSeek-FineTuning'])

for i, pred in enumerate(predictions, 1):
    print(f"\nExample {i}:")
    print(f"Prompt: {pred['prompt'].split('<｜Assistant｜>')[0].split('<｜User｜>')[1].strip()}")
    print(f"Predicted: {pred['predict']}")
    print(f"Actual: {pred['label']}")
    print(f"Match: {'✓' if pred['predict'] == pred['label'] else '✗'}")

# Calculate and show accuracy stats
def analyze_predictions(filepath, limit=None):
    """Analyze the prediction accuracy from generated_predictions.jsonl file."""
    correct = 0
    total = 0
    with open(os.path.join(filepath, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            if limit and total >= limit:
                break
            pred = json.loads(line)
            if pred['predict'] == pred['label']:
                correct += 1
            total += 1
    return correct, total, (correct / total * 100)

# Analyze a sample of predictions from each model
sample_size = 100  # Adjust as needed
print("\nPrediction Accuracy (Sample):")
for model_name, path in paths.items():
    correct, total, accuracy = analyze_predictions(path, sample_size)
    print(f"{model_name}: {correct}/{total} correct ({accuracy:.2f}%)")