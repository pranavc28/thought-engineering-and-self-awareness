"""
Create rescaled bar charts
"""

import json
import matplotlib.pyplot as plt

MODELS = ["o3", "gpt-5"]

# Load summary data for plotting
with open('thought_vs_naive_summary.json', 'r') as f:
    summary = json.load(f)

def create_rescaled_bar_chart(models, naive_f1s, overthinking_f1s, posthoc_f1s, 
                               title, ylabel, filename):
    """Create a rescaled bar chart with F1 scores displayed above bars"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    x = range(len(models))
    width = 0.25
    
    bars1 = ax.bar([i - width for i in x], naive_f1s, width, label="Naive", color="skyblue")
    bars2 = ax.bar([i for i in x], overthinking_f1s, width, label="Overthinking", color="coral")
    bars3 = ax.bar([i + width for i in x], posthoc_f1s, width, label="Automated Confidence Refinement", color="lightgreen")
    
    # Add F1 scores above bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, 1.15), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    
    # RESCALE Y-AXIS to emphasize differences
    all_scores = naive_f1s + overthinking_f1s + posthoc_f1s
    y_min = min(all_scores) - 0.02
    y_max = max(all_scores) + 0.04
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Updated {filename} (rescaled)")

print("Creating rescaled charts...\n")

# 1. Macro F1
naive_f1s = [summary[m]["naive_f1"] for m in MODELS]
overthinking_f1s = [summary[m]["overthinking_f1"] for m in MODELS]
posthoc_f1s = [summary[m]["posthoc_f1"] for m in MODELS]

create_rescaled_bar_chart(
    MODELS, naive_f1s, overthinking_f1s, posthoc_f1s,
    "Retrieval Strategy Comparison: F1 by Model",
    "Macro F1 Score",
    "thought_vs_naive_line.png"
)

# 2. NOINFO F1
naive_f1s = [summary[m]["naive_per_class"]["NOINFO"] for m in MODELS]
overthink_f1s = [summary[m]["overthinking_per_class"]["NOINFO"] for m in MODELS]
posthoc_f1s = [summary[m]["posthoc_per_class"]["NOINFO"] for m in MODELS]

create_rescaled_bar_chart(
    MODELS, naive_f1s, overthink_f1s, posthoc_f1s,
    "NOINFO Classification Performance: F1 by Model and Strategy",
    "NOINFO F1 Score",
    "noinfo_f1_comparison.png"
)

print("\n✓ All charts rescaled!")

