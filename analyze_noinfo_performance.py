"""
Analyze NOINFO Classification Performance Across Retrieval Strategies

This script analyzes how different retrieval strategies (naive, overthinking, post-hoc)
handle the NOINFO class, examining both F1 scores and confidence patterns.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

MODELS = ["o3", "o4-mini", "gpt-4o", "gpt-5-mini", "gpt-5"]

def load_logs(filepath="thought_vs_naive_logs.json"):
    """Load experiment logs"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_class_f1(predictions, golds, target_class):
    """Calculate F1 score for a specific class"""
    tp = sum(1 for p, g in zip(predictions, golds) if p == target_class and g == target_class)
    fp = sum(1 for p, g in zip(predictions, golds) if p == target_class and g != target_class)
    fn = sum(1 for p, g in zip(predictions, golds) if p != target_class and g == target_class)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def analyze_confidence_patterns(logs):
    """Analyze confidence scores for NOINFO predictions"""
    model_data = defaultdict(lambda: {
        "naive": {"correct": [], "incorrect": [], "all": []},
        "overthinking": {"correct": [], "incorrect": [], "all": []},
        "posthoc": {"correct": [], "incorrect": [], "all": []}
    })
    
    for entry in logs:
        model = entry["model"]
        gold = entry["gold"]
        
        # Naive
        naive_pred = entry["naive_label"]
        naive_conf = entry["naive_result"].get("confidence", 0.0)
        if naive_pred == "NOINFO":
            model_data[model]["naive"]["all"].append(naive_conf)
            if naive_pred == gold:
                model_data[model]["naive"]["correct"].append(naive_conf)
            else:
                model_data[model]["naive"]["incorrect"].append(naive_conf)
        
        # Overthinking
        overthink_pred = entry["overthinking_label"]
        overthink_conf = entry["overthinking_result"].get("confidence", 0.0)
        if overthink_pred == "NOINFO":
            model_data[model]["overthinking"]["all"].append(overthink_conf)
            if overthink_pred == gold:
                model_data[model]["overthinking"]["correct"].append(overthink_conf)
            else:
                model_data[model]["overthinking"]["incorrect"].append(overthink_conf)
        
        # Post-hoc
        posthoc_pred = entry["posthoc_label"]
        posthoc_conf = entry["posthoc_result"].get("confidence", 0.0)
        if posthoc_pred == "NOINFO":
            model_data[model]["posthoc"]["all"].append(posthoc_conf)
            if posthoc_pred == gold:
                model_data[model]["posthoc"]["correct"].append(posthoc_conf)
            else:
                model_data[model]["posthoc"]["incorrect"].append(posthoc_conf)
    
    return model_data

def plot_class_f1_comparison(logs, target_class, filename, title):
    """Plot F1 scores for a specific class across models and strategies"""
    model_metrics = {}
    
    for model in MODELS:
        model_logs = [l for l in logs if l["model"] == model]
        golds = [l["gold"] for l in model_logs]
        
        naive_preds = [l["naive_label"] for l in model_logs]
        overthink_preds = [l["overthinking_label"] for l in model_logs]
        posthoc_preds = [l["posthoc_label"] for l in model_logs]
        
        model_metrics[model] = {
            "naive": calculate_class_f1(naive_preds, golds, target_class),
            "overthinking": calculate_class_f1(overthink_preds, golds, target_class),
            "posthoc": calculate_class_f1(posthoc_preds, golds, target_class)
        }
    
    # Create line plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    x = range(len(MODELS))
    
    naive_f1s = [model_metrics[m]["naive"]["f1"] for m in MODELS]
    overthink_f1s = [model_metrics[m]["overthinking"]["f1"] for m in MODELS]
    posthoc_f1s = [model_metrics[m]["posthoc"]["f1"] for m in MODELS]
    
    ax.plot(x, naive_f1s, marker='o', linewidth=2.5, markersize=10, 
            label="Naive", color="#3498db")
    ax.plot(x, overthink_f1s, marker='s', linewidth=2.5, markersize=10, 
            label="Overthinking", color="#e74c3c")
    ax.plot(x, posthoc_f1s, marker='^', linewidth=2.5, markersize=10, 
            label="Post-hoc", color="#2ecc71")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"{target_class} F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"📊 {target_class} F1 comparison saved to {filename}")
    
    # Print detailed metrics
    print("\n" + "="*80)
    print(f"{target_class} Classification Metrics by Model")
    print("="*80)
    for model in MODELS:
        print(f"\n{model}:")
        for strategy in ["naive", "overthinking", "posthoc"]:
            metrics = model_metrics[model][strategy]
            print(f"  {strategy:12} - F1: {metrics['f1']:.3f}, "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f} "
                  f"(TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']})")
    
    return model_metrics


def print_confidence_summary(conf_data):
    """Print summary statistics for confidence scores"""
    print("\n" + "="*80)
    print("NOINFO Confidence Score Summary")
    print("="*80)
    
    for model in MODELS:
        print(f"\n{model}:")
        for strategy in ["naive", "overthinking", "posthoc"]:
            correct = conf_data[model][strategy]["correct"]
            incorrect = conf_data[model][strategy]["incorrect"]
            all_confs = conf_data[model][strategy]["all"]
            
            if all_confs:
                print(f"  {strategy:12}:")
                print(f"    Total NOINFO predictions: {len(all_confs)}")
                print(f"    Mean confidence (all): {np.mean(all_confs):.3f}")
                if correct:
                    print(f"    Mean confidence (correct): {np.mean(correct):.3f} (n={len(correct)})")
                if incorrect:
                    print(f"    Mean confidence (incorrect): {np.mean(incorrect):.3f} (n={len(incorrect)})")

def main():
    print("="*80)
    print("Analyzing NOINFO Classification Performance")
    print("="*80)
    
    # Load logs
    print("\nLoading logs...")
    logs = load_logs()
    print(f"Loaded {len(logs)} log entries")
    
    # Analyze confidence patterns
    print("\nAnalyzing confidence patterns...")
    conf_data = analyze_confidence_patterns(logs)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: NOINFO F1 comparison
    noinfo_metrics = plot_class_f1_comparison(
        logs, 
        "NOINFO", 
        "noinfo_f1_comparison.png",
        "NOINFO Classification Performance: F1 by Model and Strategy"
    )
    
    # Plot 2: SUPPORT F1 comparison
    support_metrics = plot_class_f1_comparison(
        logs,
        "SUPPORT",
        "support_f1_comparison.png",
        "SUPPORT Classification Performance: F1 by Model and Strategy"
    )
    
    # Plot 3: CONTRADICT F1 comparison
    contradict_metrics = plot_class_f1_comparison(
        logs,
        "CONTRADICT",
        "contradict_f1_comparison.png",
        "CONTRADICT Classification Performance: F1 by Model and Strategy"
    )
    
    # Print confidence summary
    print_confidence_summary(conf_data)
    
    # Save detailed results
    results = {
        "noinfo_f1_metrics": {
            model: {
                strategy: {
                    "f1": noinfo_metrics[model][strategy]["f1"],
                    "precision": noinfo_metrics[model][strategy]["precision"],
                    "recall": noinfo_metrics[model][strategy]["recall"],
                    "tp": noinfo_metrics[model][strategy]["tp"],
                    "fp": noinfo_metrics[model][strategy]["fp"],
                    "fn": noinfo_metrics[model][strategy]["fn"]
                }
                for strategy in ["naive", "overthinking", "posthoc"]
            }
            for model in MODELS
        },
        "support_f1_metrics": {
            model: {
                strategy: {
                    "f1": support_metrics[model][strategy]["f1"],
                    "precision": support_metrics[model][strategy]["precision"],
                    "recall": support_metrics[model][strategy]["recall"],
                    "tp": support_metrics[model][strategy]["tp"],
                    "fp": support_metrics[model][strategy]["fp"],
                    "fn": support_metrics[model][strategy]["fn"]
                }
                for strategy in ["naive", "overthinking", "posthoc"]
            }
            for model in MODELS
        },
        "contradict_f1_metrics": {
            model: {
                strategy: {
                    "f1": contradict_metrics[model][strategy]["f1"],
                    "precision": contradict_metrics[model][strategy]["precision"],
                    "recall": contradict_metrics[model][strategy]["recall"],
                    "tp": contradict_metrics[model][strategy]["tp"],
                    "fp": contradict_metrics[model][strategy]["fp"],
                    "fn": contradict_metrics[model][strategy]["fn"]
                }
                for strategy in ["naive", "overthinking", "posthoc"]
            }
            for model in MODELS
        },
        "confidence_summary": {
            model: {
                strategy: {
                    "total_noinfo_predictions": len(conf_data[model][strategy]["all"]),
                    "mean_confidence_all": float(np.mean(conf_data[model][strategy]["all"])) 
                        if conf_data[model][strategy]["all"] else 0.0,
                    "mean_confidence_correct": float(np.mean(conf_data[model][strategy]["correct"])) 
                        if conf_data[model][strategy]["correct"] else 0.0,
                    "mean_confidence_incorrect": float(np.mean(conf_data[model][strategy]["incorrect"])) 
                        if conf_data[model][strategy]["incorrect"] else 0.0,
                    "n_correct": len(conf_data[model][strategy]["correct"]),
                    "n_incorrect": len(conf_data[model][strategy]["incorrect"])
                }
                for strategy in ["naive", "overthinking", "posthoc"]
            }
            for model in MODELS
        }
    }
    
    with open("noinfo_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📝 Detailed results saved to noinfo_analysis_results.json")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. noinfo_f1_comparison.png - NOINFO F1 scores by model and strategy")
    print("  2. support_f1_comparison.png - SUPPORT F1 scores by model and strategy")
    print("  3. contradict_f1_comparison.png - CONTRADICT F1 scores by model and strategy")
    print("  4. noinfo_analysis_results.json - Detailed metrics and statistics")
    print()

if __name__ == "__main__":
    main()

