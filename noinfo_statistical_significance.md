# NOINFO Classification - Statistical Significance

McNemar's test for paired classifier comparison (200 claims).
Tests whether differences in NOINFO classification are statistically significant.

## Within-Model Comparisons

| Model | Comparison | χ² Statistic | p-value | Significant? |
|-------|------------|--------------|---------|-------------|
| o3 | Naive vs Overthinking | 0.893 | 0.6894 | No |
| o3 | Naive vs Auto Refine | 2.703 | 0.2004 | No |
| o3 | Overthinking vs Auto Refine | 0.516 | 0.9450 | No |
| o4-mini | Naive vs Overthinking | 0.893 | 0.6894 | No |
| o4-mini | Naive vs Auto Refine | 0.485 | 0.9725 | No |
| o4-mini | Overthinking vs Auto Refine | 0.000 | 1.0000 | No |
| gpt-4o | Naive vs Overthinking | 0.160 | 1.0000 | No |
| gpt-4o | Naive vs Auto Refine | 0.042 | 1.0000 | No |
| gpt-4o | Overthinking vs Auto Refine | 0.000 | 1.0000 | No |
| gpt-5-mini | Naive vs Overthinking | 0.300 | 1.0000 | No |
| gpt-5-mini | Naive vs Auto Refine | 0.000 | 1.0000 | No |
| gpt-5-mini | Overthinking vs Auto Refine | 0.114 | 1.0000 | No |
| gpt-5 | Naive vs Overthinking | 1.250 | 0.5271 | No |
| gpt-5 | Naive vs Auto Refine | 0.450 | 1.0000 | No |
| gpt-5 | Overthinking vs Auto Refine | 3.115 | 0.1551 | No |

## Cross-Model Comparisons

| Comparison | χ² Statistic | p-value | Significant? |
|------------|--------------|---------|-------------|
| GPT-5 Auto Refine vs O3 Naive | 0.000 | 1.0000 | No |

*p < 0.05 indicates statistically significant difference (marked with **)*

**Interpretation:** McNemar's test examines disagreements between classifiers. A significant result means one classifier consistently outperforms the other, not just by random chance.
