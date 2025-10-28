"""
Statistical Significance for All Models - Within and Cross-Model Comparisons
Using permutation tests (more powerful for small samples)
"""

import json
import numpy as np

# Load logs
with open('thought_vs_naive_logs.json', 'r') as f:
    all_logs = json.load(f)

def permutation_test(predictions1, predictions2, golds, n_permutations=10000):
    """
    Permutation test for paired classifiers
    Returns (observed_diff, p_value, accuracy1, accuracy2)
    """
    # Observed difference in accuracy
    acc1 = sum(1 for p, g in zip(predictions1, golds) if p == g) / len(golds)
    acc2 = sum(1 for p, g in zip(predictions2, golds) if p == g) / len(golds)
    observed_diff = acc1 - acc2
    
    # Generate null distribution by random swapping
    count_as_extreme = 0
    
    for _ in range(n_permutations):
        perm_acc1_correct = 0
        perm_acc2_correct = 0
        
        for p1, p2, g in zip(predictions1, predictions2, golds):
            # Randomly swap with 50% probability
            if np.random.random() < 0.5:
                p1, p2 = p2, p1
            
            if p1 == g:
                perm_acc1_correct += 1
            if p2 == g:
                perm_acc2_correct += 1
        
        perm_diff = (perm_acc1_correct - perm_acc2_correct) / len(golds)
        
        if abs(perm_diff) >= abs(observed_diff):
            count_as_extreme += 1
    
    p_value = count_as_extreme / n_permutations
    return observed_diff, p_value, acc1, acc2

print("="*80)
print("COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*80)

results = []

# ==================== WITHIN-MODEL COMPARISONS ====================
models_to_test = ["o3", "gpt-5"]

for model_name in models_to_test:
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - WITHIN-MODEL COMPARISONS")
    print(f"{'='*80}")
    
    model_logs = [l for l in all_logs if l["model"] == model_name]
    model_golds = [l["gold"] for l in model_logs]
    
    model_naive = [l["naive_label"] for l in model_logs]
    model_overthinking = [l["overthinking_label"] for l in model_logs]
    model_auto_refine = [l["posthoc_label"] for l in model_logs]
    
    # 1. Overthinking vs Automated Refinement
    print("\n" + "─"*80)
    print(f"Overthinking vs Automated Refinement")
    print("─"*80)
    obs_diff, p_val, acc1, acc2 = permutation_test(model_overthinking, model_auto_refine, model_golds)
    confidence = 1 - p_val
    sig = "✓ SIGNIFICANT" if p_val < 0.05 else "Not significant"
    print(f"Overthinking: {acc1:.4f} ({acc1*100:.2f}%) | Auto Refine: {acc2:.4f} ({acc2*100:.2f}%) | Diff: {obs_diff:.4f}")
    print(f"p-value: {p_val:.4f}, Confidence: {confidence:.4f} ({confidence*100:.2f}%) [{sig}]")
    
    results.append({
        "model": model_name,
        "comparison": "Overthinking vs Automated Refinement",
        "sample_size": len(model_golds),
        "p_value": p_val,
        "confidence": confidence,
        "acc1": acc1,
        "acc2": acc2,
        "diff": obs_diff
    })
    
    # 2. Automated Refinement vs Naive
    print("\n" + "─"*80)
    print(f"Automated Refinement vs Naive")
    print("─"*80)
    obs_diff, p_val, acc1, acc2 = permutation_test(model_auto_refine, model_naive, model_golds)
    confidence = 1 - p_val
    sig = "✓ SIGNIFICANT" if p_val < 0.05 else "Not significant"
    print(f"Auto Refine: {acc1:.4f} ({acc1*100:.2f}%) | Naive: {acc2:.4f} ({acc2*100:.2f}%) | Diff: {obs_diff:.4f}")
    print(f"p-value: {p_val:.4f}, Confidence: {confidence:.4f} ({confidence*100:.2f}%) [{sig}]")
    
    results.append({
        "model": model_name,
        "comparison": "Automated Refinement vs Naive",
        "sample_size": len(model_golds),
        "p_value": p_val,
        "confidence": confidence,
        "acc1": acc1,
        "acc2": acc2,
        "diff": obs_diff
    })
    
    # 3. Overthinking vs Naive
    print("\n" + "─"*80)
    print(f"Overthinking vs Naive")
    print("─"*80)
    obs_diff, p_val, acc1, acc2 = permutation_test(model_overthinking, model_naive, model_golds)
    confidence = 1 - p_val
    sig = "✓ SIGNIFICANT" if p_val < 0.05 else "Not significant"
    print(f"Overthinking: {acc1:.4f} ({acc1*100:.2f}%) | Naive: {acc2:.4f} ({acc2*100:.2f}%) | Diff: {obs_diff:.4f}")
    print(f"p-value: {p_val:.4f}, Confidence: {confidence:.4f} ({confidence*100:.2f}%) [{sig}]")
    
    results.append({
        "model": model_name,
        "comparison": "Overthinking vs Naive",
        "sample_size": len(model_golds),
        "p_value": p_val,
        "confidence": confidence,
        "acc1": acc1,
        "acc2": acc2,
        "diff": obs_diff
    })

# ==================== SUMMARY TABLE ====================
print("\n" + "="*80)
print("SUMMARY TABLE - ALL COMPARISONS")
print("="*80 + "\n")

print(f"{'Model':<12} | {'Comparison':<45} | {'Sample':<6} | {'Confidence':<15} | Sig?")
print("-" * 100)

for r in results:
    sig_text = "✓ Yes" if r["p_value"] < 0.05 else "No"
    model_display = r["model"] if r["model"] != "cross-model" else "Cross"
    print(f"{model_display:<12} | {r['comparison']:<45} | {r['sample_size']:<6} | "
          f"{r['confidence']:.4f} ({r['confidence']*100:.1f}%) | {sig_text}")

# Save to files
with open("comprehensive_statistical_significance.md", "w") as f:
    f.write("# Comprehensive Statistical Significance Analysis\n\n")
    f.write("**Method**: Permutation Test (10,000 permutations)\n\n")
    f.write("**Metric**: NOINFO Classification Accuracy\n\n")
    
    f.write("## Summary Table\n\n")
    f.write("| Model | Comparison | Sample Size | Confidence | Significant? |\n")
    f.write("|-------|------------|-------------|------------|-------------|\n")
    
    for r in results:
        sig_marker = " **" if r["p_value"] < 0.05 else ""
        sig_text = "✓ Yes" if r["p_value"] < 0.05 else "No"
        f.write(f"| {r['model']} | {r['comparison']} | {r['sample_size']} | "
                f"{r['confidence']:.4f} ({r['confidence']*100:.1f}%){sig_marker} | {sig_text} |\n")
    
    f.write("\n*Confidence = 1 - p-value. Values marked with ** are statistically significant (p < 0.05)*\n")

with open("comprehensive_statistical_significance.json", "w") as jf:
    json.dump({
        "method": "Permutation Test (10,000 permutations)",
        "metric": "NOINFO Classification Accuracy",
        "comparisons": results
    }, jf, indent=2)

print("\n" + "="*80)
print("✓ Results saved to: comprehensive_statistical_significance.md")
print("✓ Results saved to: comprehensive_statistical_significance.json")
print("="*80)

