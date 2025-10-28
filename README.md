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

### 5. Create Rescaled Visualizations
```bash
python create_rescaled_charts_with_stats.py
```

Generates rescaled bar charts that emphasize performance differences between strategies:
- Creates separate NOINFO F1 charts for each model (o3, gpt-5)
- Rescales y-axis to highlight small but meaningful differences
- Displays exact F1 scores above each bar

Output: `noinfo_f1_o3.png`, `noinfo_f1_gpt-5.png`

**Why rescaling?** Small F1 improvements (0.02-0.05) can be statistically significant but hard to see on a [0,1] scale. Rescaled charts make these differences visible.

### 6. Run Statistical Significance Tests
```bash
python focused_statsig_comparisons_v2.py
```

Performs comprehensive statistical testing using permutation tests (10,000 permutations):
- **Within-model comparisons**: Tests if strategy differences are statistically significant
- Compares: Overthinking vs Automated Refinement, Automated Refinement vs Naive, Overthinking vs Naive
- Reports p-values, confidence intervals, and significance markers

Output: `comprehensive_statistical_significance.md`, `comprehensive_statistical_significance.json`, `statistical_significance_tables.md`

**Key insight**: p < 0.05 indicates statistically significant differences between strategies

**Expected improvement**: +0.01 to +0.03 F1

## Files
- `thought_vs_naive_scifact.py` (527 lines) - Main experiment
- `optimize_thresholds.py` (222 lines) - Threshold optimizer
- `create_rescaled_charts_with_stats.py` - Rescaled bar chart generator
- `focused_statsig_comparisons_v2.py` - Statistical significance testing (permutation tests)
- `data/` - SciFact dataset
- `threshold_config.json` - Optimized thresholds per model
- `thought_vs_naive_logs.json` - Detailed experiment logs
- `thought_vs_naive_summary.json` - Aggregate F1 scores
- `noinfo_f1_o3.png`, `noinfo_f1_gpt-5.png` - Rescaled NOINFO performance charts
- `comprehensive_statistical_significance.md` - Statistical test results
- `comprehensive_statistical_significance.json` - Statistical test data
- `statistical_significance_tables.md` - McNemar's test tables
