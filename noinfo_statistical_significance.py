"""
Statistical Significance Testing for NOINFO Classification
- Within-model comparisons (e.g., gpt-5 overthinking vs gpt-5 automated refinement)
- Cross-model comparisons (e.g., gpt-5 automated refinement vs o3 naive)
"""

import json
from math import sqrt, erf

MODELS = ["o3", "o4-mini", "gpt-4o", "gpt-5-mini", "gpt-5"]

# Load logs
with open('thought_vs_naive_logs.json', 'r') as f:
    all_logs = json.load(f)

def chi2_cdf(x, df=1):
    """Cumulative distribution function for chi-square with df=1"""
    if x <= 0:
        return 0.0
    return erf(sqrt(x / 2.0))

def mcnemar_test(predictions1, predictions2, golds):
    """
    Perform McNemar's test for NOINFO classification
    Returns (statistic, p-value, contingency details)
    """
    # Count discordant pairs
    p1_correct_p2_wrong = sum(1 for p1, p2, g in zip(predictions1, predictions2, golds) 
                             if p1 == g and p2 != g)
    p1_wrong_p2_correct = sum(1 for p1, p2, g in zip(predictions1, predictions2, golds) 
                             if p1 != g and p2 == g)
    both_correct = sum(1 for p1, p2, g in zip(predictions1, predictions2, golds) 
                      if p1 == g and p2 == g)
    both_wrong = sum(1 for p1, p2, g in zip(predictions1, predictions2, golds) 
                    if p1 != g and p2 != g)
    
    # If no discordant pairs, models are identical
    if p1_correct_p2_wrong + p1_wrong_p2_correct == 0:
        return 0.0, 1.0, (both_correct, p1_correct_p2_wrong, p1_wrong_p2_correct, both_wrong)
    
    # McNemar's test statistic with continuity correction
    n = p1_correct_p2_wrong + p1_wrong_p2_correct
    statistic = ((abs(p1_correct_p2_wrong - p1_wrong_p2_correct) - 1) ** 2) / n
    
    # Approximate p-value
    p_value = 2 * (1 - chi2_cdf(statistic))
    
    # Clamp p-value
    p_value = max(0.0, min(1.0, p_value))
    
    return statistic, p_value, (both_correct, p1_correct_p2_wrong, p1_wrong_p2_correct, both_wrong)

def compute_noinfo_accuracy(predictions, golds):
    """Compute accuracy for NOINFO prediction"""
    correct = sum(1 for p, g in zip(predictions, golds) if p == g)
    return correct / len(golds)

print("="*80)
print("NOINFO Classification - Statistical Significance Testing")
print("="*80)

# Within-model comparisons
print("\n### WITHIN-MODEL COMPARISONS (Same Model, Different Strategies) ###\n")

within_model_results = []

for model in MODELS:
    print(f"\n{model.upper()}")
    print("-" * 40)
    
    model_logs = [l for l in all_logs if l["model"] == model]
    golds = [l["gold"] for l in model_logs]
    
    # Get predictions
    naive_preds = [l["naive_label"] for l in model_logs]
    overthinking_preds = [l["overthinking_label"] for l in model_logs]
    posthoc_preds = [l["posthoc_label"] for l in model_logs]
    
    # Compute accuracies
    naive_acc = compute_noinfo_accuracy(naive_preds, golds)
    overthinking_acc = compute_noinfo_accuracy(overthinking_preds, golds)
    posthoc_acc = compute_noinfo_accuracy(posthoc_preds, golds)
    
    print(f"Accuracies: Naive={naive_acc:.3f}, Overthinking={overthinking_acc:.3f}, Auto Refine={posthoc_acc:.3f}")
    
    # Naive vs Overthinking
    stat, pval, contingency = mcnemar_test(naive_preds, overthinking_preds, golds)
    bc, p1c_p2w, p1w_p2c, bw = contingency
    sig = "✓ SIGNIFICANT" if pval < 0.05 else "Not significant"
    print(f"\nNaive vs Overthinking:")
    print(f"  χ² = {stat:.3f}, p = {pval:.4f} [{sig}]")
    print(f"  Both correct: {bc}, Naive better: {p1c_p2w}, Overthinking better: {p1w_p2c}, Both wrong: {bw}")
    
    within_model_results.append({
        "model": model,
        "comparison": "Naive vs Overthinking",
        "statistic": stat,
        "p_value": pval,
        "significant": pval < 0.05
    })
    
    # Naive vs Auto Refine
    stat, pval, contingency = mcnemar_test(naive_preds, posthoc_preds, golds)
    bc, p1c_p2w, p1w_p2c, bw = contingency
    sig = "✓ SIGNIFICANT" if pval < 0.05 else "Not significant"
    print(f"\nNaive vs Automated Refinement:")
    print(f"  χ² = {stat:.3f}, p = {pval:.4f} [{sig}]")
    print(f"  Both correct: {bc}, Naive better: {p1c_p2w}, Auto Refine better: {p1w_p2c}, Both wrong: {bw}")
    
    within_model_results.append({
        "model": model,
        "comparison": "Naive vs Auto Refine",
        "statistic": stat,
        "p_value": pval,
        "significant": pval < 0.05
    })
    
    # Overthinking vs Auto Refine
    stat, pval, contingency = mcnemar_test(overthinking_preds, posthoc_preds, golds)
    bc, p1c_p2w, p1w_p2c, bw = contingency
    sig = "✓ SIGNIFICANT" if pval < 0.05 else "Not significant"
    print(f"\nOverthinking vs Automated Refinement:")
    print(f"  χ² = {stat:.3f}, p = {pval:.4f} [{sig}]")
    print(f"  Both correct: {bc}, Overthinking better: {p1c_p2w}, Auto Refine better: {p1w_p2c}, Both wrong: {bw}")
    
    within_model_results.append({
        "model": model,
        "comparison": "Overthinking vs Auto Refine",
        "statistic": stat,
        "p_value": pval,
        "significant": pval < 0.05
    })

# Cross-model comparisons
print("\n\n### CROSS-MODEL COMPARISONS ###\n")

cross_model_results = []

# Key comparison: gpt-5 Auto Refine vs o3 Naive
print("GPT-5 Automated Refinement vs O3 Naive")
print("-" * 40)

gpt5_logs = [l for l in all_logs if l["model"] == "gpt-5"]
o3_logs = [l for l in all_logs if l["model"] == "o3"]

# Match by claim text (more robust than claim_id)
claim_to_gpt5 = {l["claim"]: l for l in gpt5_logs}
claim_to_o3 = {l["claim"]: l for l in o3_logs}

# Find common claims
common_claims = set(claim_to_gpt5.keys()) & set(claim_to_o3.keys())
print(f"Found {len(common_claims)} common claims between GPT-5 and O3")

if len(common_claims) > 0:
    # Extract aligned predictions
    golds = []
    gpt5_auto_preds = []
    o3_naive_preds = []
    
    for claim in sorted(common_claims):
        golds.append(claim_to_gpt5[claim]["gold"])
        gpt5_auto_preds.append(claim_to_gpt5[claim]["posthoc_label"])
        o3_naive_preds.append(claim_to_o3[claim]["naive_label"])
    
    gpt5_acc = compute_noinfo_accuracy(gpt5_auto_preds, golds)
    o3_acc = compute_noinfo_accuracy(o3_naive_preds, golds)
    
    print(f"Accuracies: GPT-5 Auto Refine={gpt5_acc:.3f}, O3 Naive={o3_acc:.3f}")
    
    stat, pval, contingency = mcnemar_test(gpt5_auto_preds, o3_naive_preds, golds)
    bc, p1c_p2w, p1w_p2c, bw = contingency
    sig = "✓ SIGNIFICANT" if pval < 0.05 else "Not significant"
    print(f"\nχ² = {stat:.3f}, p = {pval:.4f} [{sig}]")
    print(f"Both correct: {bc}, GPT-5 better: {p1c_p2w}, O3 better: {p1w_p2c}, Both wrong: {bw}")
    
    cross_model_results.append({
        "comparison": "GPT-5 Auto Refine vs O3 Naive",
        "statistic": stat,
        "p_value": pval,
        "significant": pval < 0.05
    })
else:
    print("ERROR: No common claims found between models")

# Additional cross-model: Best vs Best
print("\n\nBest Performer Comparison")
print("-" * 40)
print("GPT-5 Automated Refinement vs O3 Naive (best of each generation)")

# Save results to markdown
with open("noinfo_statistical_significance.md", "w") as f:
    f.write("# NOINFO Classification - Statistical Significance\n\n")
    f.write("McNemar's test for paired classifier comparison (200 claims).\n")
    f.write("Tests whether differences in NOINFO classification are statistically significant.\n\n")
    
    f.write("## Within-Model Comparisons\n\n")
    f.write("| Model | Comparison | χ² Statistic | p-value | Significant? |\n")
    f.write("|-------|------------|--------------|---------|-------------|\n")
    
    for result in within_model_results:
        sig = "✓ **Yes**" if result["significant"] else "No"
        marker = " **" if result["significant"] else ""
        f.write(f"| {result['model']} | {result['comparison']} | "
                f"{result['statistic']:.3f} | {result['p_value']:.4f}{marker} | {sig} |\n")
    
    f.write("\n## Cross-Model Comparisons\n\n")
    f.write("| Comparison | χ² Statistic | p-value | Significant? |\n")
    f.write("|------------|--------------|---------|-------------|\n")
    
    for result in cross_model_results:
        sig = "✓ **Yes**" if result["significant"] else "No"
        marker = " **" if result["significant"] else ""
        f.write(f"| {result['comparison']} | {result['statistic']:.3f} | "
                f"{result['p_value']:.4f}{marker} | {sig} |\n")
    
    f.write("\n*p < 0.05 indicates statistically significant difference (marked with **)*\n")
    f.write("\n**Interpretation:** McNemar's test examines disagreements between classifiers. ")
    f.write("A significant result means one classifier consistently outperforms the other, ")
    f.write("not just by random chance.\n")

print("\n\n" + "="*80)
print("✓ Results saved to: noinfo_statistical_significance.md")
print("="*80)

