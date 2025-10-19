"""
Automated Threshold Optimization via Grid Search
Tests all threshold combinations and finds optimal values by F1 score.
"""

import json
from itertools import product

LABELS = {"SUPPORT", "CONTRADICT", "NOINFO"}

def compute_f1_macro(predictions, golds):
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in LABELS}
    for pred, gold in zip(predictions, golds):
        for lab in LABELS:
            if pred == lab and gold == lab:
                counts[lab]["tp"] += 1
            elif pred == lab and gold != lab:
                counts[lab]["fp"] += 1
            elif pred != lab and gold == lab:
                counts[lab]["fn"] += 1
    
    f1_scores = {}
    for lab in LABELS:
        tp, fp, fn = counts[lab]["tp"], counts[lab]["fp"], counts[lab]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores[lab] = f1
    
    return sum(f1_scores.values()) / len(f1_scores)

def apply_classification_threshold(label, confidence, threshold):
    """Convert low-confidence non-NOINFO to NOINFO"""
    if confidence < threshold and label != "NOINFO":
        return "NOINFO"
    return label

def simulate_posthoc_with_threshold(log, threshold):
    """Simulate what would happen with different posthoc threshold"""
    metadata = log.get("posthoc_metadata", {})
    initial_conf = metadata.get("initial_confidence", 0.5)
    refinement_triggered = metadata.get("refinement_triggered", False)
    
    # If confidence >= threshold, refinement wouldn't have triggered
    # So use naive result instead of posthoc result
    if initial_conf >= threshold:
        return log["naive_label"]  # Would not have refined
    else:
        return log["posthoc_label"]  # Would have refined

def grid_search():
    with open('thought_vs_naive_logs.json', 'r') as f:
        logs = json.load(f)
    
    # Check for required fields
    sample = logs[0]
    has_results = 'naive_result' in sample
    has_posthoc_metadata = 'posthoc_metadata' in sample
    
    if not has_results:
        print("ERROR: Logs missing confidence scores. Re-run experiments with updated code.")
        return
    
    if not has_posthoc_metadata:
        print("WARNING: Logs missing posthoc_metadata. Can't optimize posthoc thresholds.")
        has_posthoc_metadata = False
    
    # Grid search ranges
    classification_thresholds = [round(x * 0.05, 2) for x in range(10, 16)]  # 0.50-0.75
    posthoc_thresholds = [round(x * 0.05, 2) for x in range(14, 20)]  # 0.70-0.95
    
    print("=" * 80)
    print("AUTOMATED THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"\nTesting classification thresholds: {classification_thresholds}")
    if has_posthoc_metadata:
        print(f"Testing posthoc thresholds: {posthoc_thresholds}")
    
    models = sorted(set(l["model"] for l in logs))
    best_configs = {}
    
    for model in models:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model}")
        print(f"{'=' * 80}")
        
        model_logs = [l for l in logs if l["model"] == model]
        golds = [l["gold"] for l in model_logs]
        
        # Baseline
        print("\nBaseline (no threshold):")
        for approach in ["naive", "overthinking", "posthoc"]:
            preds = [l[f"{approach}_label"] for l in model_logs]
            f1 = compute_f1_macro(preds, golds)
            print(f"  {approach:12s}: F1 = {f1:.4f}")
        
        best_configs[model] = {}
        
        # 1. Grid search classification thresholds
        for approach in ["naive", "overthinking", "posthoc"]:
            print(f"\n{approach.upper()}: Testing classification thresholds...")
            
            best_f1, best_threshold = 0.0, None
            for threshold in classification_thresholds:
                predictions = []
                for log in model_logs:
                    result = log.get(f"{approach}_result", {})
                    if not result:
                        predictions.append(log[f"{approach}_label"])
                        continue
                    label = result.get("label", "NOINFO")
                    confidence = float(result.get("confidence", 1.0))
                    predictions.append(apply_classification_threshold(label, confidence, threshold))
                
                f1 = compute_f1_macro(predictions, golds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            best_configs[model][f"{approach}_classification"] = {
                "threshold": best_threshold,
                "f1": best_f1
            }
            print(f"  ✓ Best: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
        
        # 2. Grid search posthoc refinement threshold
        if has_posthoc_metadata:
            print(f"\nPOSTHOC REFINEMENT: Testing thresholds...")
            
            best_f1, best_posthoc_threshold = 0.0, None
            for threshold in posthoc_thresholds:
                predictions = []
                for log in model_logs:
                    predictions.append(simulate_posthoc_with_threshold(log, threshold))
                
                f1 = compute_f1_macro(predictions, golds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_posthoc_threshold = threshold
            
            best_configs[model]["posthoc_refinement"] = {
                "threshold": best_posthoc_threshold,
                "f1": best_f1
            }
            print(f"  ✓ Best: threshold={best_posthoc_threshold:.2f}, F1={best_f1:.4f}")
    
    # Output optimal config
    print(f"\n{'=' * 80}")
    print("OPTIMAL CONFIGURATION (Copy-paste into thought_vs_naive_scifact.py)")
    print(f"{'=' * 80}")
    
    # Build simplified output
    simplified_config = {}
    for model in models:
        config = best_configs[model]
        
        # Use best approach's classification threshold
        best_approach = max(["naive", "overthinking", "posthoc"], 
                           key=lambda a: config[f"{a}_classification"]["f1"])
        class_threshold = config[f"{best_approach}_classification"]["threshold"]
        posthoc_threshold = config.get("posthoc_refinement", {}).get("threshold", 0.80)
        
        simplified_config[model] = {
            "classification": class_threshold,
            "posthoc_refinement": posthoc_threshold
        }
    
    print("\nTHRESHOLDS = {")
    for model, thresholds in simplified_config.items():
        print(f'    "{model}": {{')
        print(f'        "classification": {thresholds["classification"]:.2f},')
        print(f'        "posthoc_refinement": {thresholds["posthoc_refinement"]:.2f},')
        print(f'    }},')
    print("}")
    print(f"\nDEFAULT_THRESHOLDS = {{\"classification\": 0.55, \"posthoc_refinement\": 0.75}}")
    
    # Show improvements
    print(f"\n{'=' * 80}")
    print("EXPECTED IMPROVEMENTS")
    print(f"{'=' * 80}")
    
    for model in models:
        print(f"\n{model}:")
        config = best_configs[model]
        model_logs = [l for l in logs if l["model"] == model]
        golds = [l["gold"] for l in model_logs]
        
        for approach in ["naive", "overthinking", "posthoc"]:
            baseline_preds = [l[f"{approach}_label"] for l in model_logs]
            baseline_f1 = compute_f1_macro(baseline_preds, golds)
            optimized_f1 = config[f"{approach}_classification"]["f1"]
            improvement = optimized_f1 - baseline_f1
            status = "✓" if improvement > 0.001 else "→"
            print(f"  {approach:12s}: {baseline_f1:.4f} → {optimized_f1:.4f} ({improvement:+.4f}) {status}")
        
        if has_posthoc_metadata:
            posthoc_baseline = compute_f1_macro([l["posthoc_label"] for l in model_logs], golds)
            posthoc_optimized = config["posthoc_refinement"]["f1"]
            posthoc_improvement = posthoc_optimized - posthoc_baseline
            status = "✓" if posthoc_improvement > 0.001 else "→"
            print(f"  posthoc+refine: {posthoc_baseline:.4f} → {posthoc_optimized:.4f} ({posthoc_improvement:+.4f}) {status}")
    
    # Save detailed results
    with open('threshold_recommendations_full.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Save simplified config (ready to use)
    with open('threshold_config.json', 'w') as f:
        output = {
            "THRESHOLDS": simplified_config,
            "DEFAULT_THRESHOLDS": {"classification": 0.55, "posthoc_refinement": 0.75}
        }
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("✓ Saved full details to: threshold_recommendations_full.json")
    print("✓ Saved copy-paste config to: threshold_config.json")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    grid_search()
