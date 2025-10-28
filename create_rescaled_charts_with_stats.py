"""
Create rescaled bar charts
"""

import json
import matplotlib.pyplot as plt

MODELS = ["o3", "gpt-5"]

# Load summary data for plotting
with open('thought_vs_naive_summary.json', 'r') as f:
    summary = json.load(f)

def create_rescaled_bar_chart(strategies, f1_scores, title, ylabel, filename):
    """Create a rescaled bar chart with F1 scores displayed above bars for a single model"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    x = range(len(strategies))
    width = 0.6
    
    colors = ["skyblue", "coral", "lightgreen"]
    bars = ax.bar(x, f1_scores, width, color=colors)
    
    # Add F1 scores above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Strategy - least complex (left) to most complex (right)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis="y", alpha=0.3)
    
    # RESCALE Y-AXIS to emphasize differences
    y_min = min(f1_scores) - 0.02
    y_max = max(f1_scores) + 0.04
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Created {filename} (rescaled)")

print("Creating rescaled NOINFO F1 charts...\n")

# Create separate NOINFO F1 chart for each model
strategies = ["Naive", "Overthinking", "Automated Confidence Refinement"]

for model in MODELS:
    naive_f1 = summary[model]["naive_per_class"]["NOINFO"]
    overthinking_f1 = summary[model]["overthinking_per_class"]["NOINFO"]
    posthoc_f1 = summary[model]["posthoc_per_class"]["NOINFO"]
    
    f1_scores = [naive_f1, overthinking_f1, posthoc_f1]
    
    create_rescaled_bar_chart(
        strategies,
        f1_scores,
        f"NOINFO Classification Performance: {model.upper()}",
        "NOINFO F1 Score",
        f"noinfo_f1_{model}.png"
    )

print("\n✓ All NOINFO charts created!")

