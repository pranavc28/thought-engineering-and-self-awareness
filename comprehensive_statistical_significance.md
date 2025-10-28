# Comprehensive Statistical Significance Analysis

**Method**: Permutation Test (10,000 permutations)

**Metric**: NOINFO Classification Accuracy

## Summary Table

| Model | Comparison | Sample Size | Confidence | Significant? |
|-------|------------|-------------|------------|-------------|
| o3 | Overthinking vs Automated Refinement | 200 | 0.8261 (82.6%) | No |
| o3 | Automated Refinement vs Naive | 200 | 0.9348 (93.5%) | No |
| o3 | Overthinking vs Naive | 200 | 0.9896 (99.0%) ** | ✓ Yes |
| gpt-5 | Overthinking vs Automated Refinement | 200 | 0.6103 (61.0%) | No |
| gpt-5 | Automated Refinement vs Naive | 200 | 0.9535 (95.3%) ** | ✓ Yes |
| gpt-5 | Overthinking vs Naive | 200 | 0.8711 (87.1%) | No |

*Confidence = 1 - p-value. Values marked with ** are statistically significant (p < 0.05)*
