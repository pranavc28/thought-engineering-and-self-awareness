# SciFact Retrieval + Threshold Optimization

Automated threshold tuning for retrieval strategies (NAIVE vs OVERTHINKING vs POST-HOC).

## Quick Start

### 1. Run Experiments
```bash
export OPENAI_API_KEY=your_key
python thought_vs_naive_scifact.py
```

Compares 3 retrieval strategies:
- **NAIVE**: Direct query generation
- **OVERTHINKING**: Pre-reasoning + adaptive search
- **POST-HOC**: Retrieve → assess → refine if needed

Output: `thought_vs_naive_logs.json`, `thought_vs_naive_summary.json`, `thought_vs_naive_results.png`

### 2. Optimize Thresholds
```bash
python optimize_thresholds.py
```

Grid searches all threshold combinations, finds optimal by F1 score.

Optimizes:
- Classification threshold (converts low-confidence → NOINFO)
- Posthoc refinement threshold (when to retrieve more papers)

### 3. Apply Optimized Thresholds

Copy output into `thought_vs_naive_scifact.py`:

```python
THRESHOLDS = {
    "o3": {
        "classification": 0.60,      # From optimizer
        "posthoc_refinement": 0.85,  # From optimizer
    },
    "gpt-4o": {
        "classification": 0.58,
        "posthoc_refinement": 0.80,
    },
}
```

### 4. Re-run & Compare
```bash
python thought_vs_naive_scifact.py
```

Check if F1 scores improved.

### 5. Analyze NOINFO Performance
```bash
python analyze_noinfo_performance.py
```

Deep-dive analysis of per-class classification performance:
- **Graph 1**: NOINFO F1 scores by model and strategy (line plot comparing all three)
- **Graph 2**: SUPPORT F1 scores by model and strategy (line plot comparing all three)
- **Graph 3**: CONTRADICT F1 scores by model and strategy (line plot comparing all three)

Output: `noinfo_f1_comparison.png`, `support_f1_comparison.png`, `contradict_f1_comparison.png`, `noinfo_analysis_results.json`

This helps diagnose why certain strategies struggle with NOINFO predictions by examining:
- F1, precision, and recall for NOINFO class
- Confidence score distributions for correct vs incorrect predictions
- Per-model differences in confidence calibration

## Configuration

```python
SAMPLES = 30           # Number of test claims
MODELS = ["o3", "gpt-4o", "gpt-5-mini", "gpt-5"]
MAX_WORKERS = 20       # Parallel processes
```

## How It Works

1. Experiments generate results with confidence scores
2. Optimizer grid searches all thresholds, selects best by F1
3. Apply optimized thresholds
4. Re-run → improved performance

**Expected improvement**: +0.01 to +0.03 F1

## Files
- `thought_vs_naive_scifact.py` (527 lines) - Main experiment
- `optimize_thresholds.py` (222 lines) - Threshold optimizer
- `analyze_noinfo_performance.py` - NOINFO classification deep-dive
- `data/` - SciFact dataset
- `threshold_config.json` - Optimized thresholds per model
- `thought_vs_naive_logs.json` - Detailed experiment logs
- `thought_vs_naive_summary.json` - Aggregate F1 scores
- `noinfo_analysis_results.json` - NOINFO-specific metrics
