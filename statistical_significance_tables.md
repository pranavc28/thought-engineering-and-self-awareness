# Statistical Significance Testing

McNemar's test compares paired classifiers on the same test set (200 claims).
It tests whether the disagreements between two classifiers are systematic or due to chance.


### Overall Macro F1 - Statistical Significance (McNemar's Test)

| Model | Comparison | χ² Statistic | p-value | Significant? |
|-------|------------|--------------|---------|-------------|
| o3 | Naive vs Overthinking | 6.261 | 0.0247 ** | ✓ Yes |
| o3 | Naive vs Auto Conf Refine | 3.368 | 0.1329 | No |
| o3 | Overthinking vs Auto Conf Refine | 0.643 | 0.8454 | No |
| gpt-5 | Naive vs Overthinking | 1.136 | 0.5728 | No |
| gpt-5 | Naive vs Auto Conf Refine | 2.560 | 0.2192 | No |
| gpt-5 | Overthinking vs Auto Conf Refine | 0.190 | 1.0000 | No |

*p < 0.05 indicates statistically significant difference (marked with **)*

### NOINFO F1 - Statistical Significance (McNemar's Test)

| Model | Comparison | χ² Statistic | p-value | Significant? |
|-------|------------|--------------|---------|-------------|
| o3 | Naive vs Overthinking | 6.261 | 0.0247 ** | ✓ Yes |
| o3 | Naive vs Auto Conf Refine | 3.368 | 0.1329 | No |
| o3 | Overthinking vs Auto Conf Refine | 0.643 | 0.8454 | No |
| gpt-5 | Naive vs Overthinking | 1.136 | 0.5728 | No |
| gpt-5 | Naive vs Auto Conf Refine | 2.560 | 0.2192 | No |
| gpt-5 | Overthinking vs Auto Conf Refine | 0.190 | 1.0000 | No |

*p < 0.05 indicates statistically significant difference (marked with **)*

### SUPPORT F1 - Statistical Significance (McNemar's Test)

| Model | Comparison | χ² Statistic | p-value | Significant? |
|-------|------------|--------------|---------|-------------|
| o3 | Naive vs Overthinking | 6.261 | 0.0247 ** | ✓ Yes |
| o3 | Naive vs Auto Conf Refine | 3.368 | 0.1329 | No |
| o3 | Overthinking vs Auto Conf Refine | 0.643 | 0.8454 | No |
| gpt-5 | Naive vs Overthinking | 1.136 | 0.5728 | No |
| gpt-5 | Naive vs Auto Conf Refine | 2.560 | 0.2192 | No |
| gpt-5 | Overthinking vs Auto Conf Refine | 0.190 | 1.0000 | No |

*p < 0.05 indicates statistically significant difference (marked with **)*

### CONTRADICT F1 - Statistical Significance (McNemar's Test)

| Model | Comparison | χ² Statistic | p-value | Significant? |
|-------|------------|--------------|---------|-------------|
| o3 | Naive vs Overthinking | 6.261 | 0.0247 ** | ✓ Yes |
| o3 | Naive vs Auto Conf Refine | 3.368 | 0.1329 | No |
| o3 | Overthinking vs Auto Conf Refine | 0.643 | 0.8454 | No |
| gpt-5 | Naive vs Overthinking | 1.136 | 0.5728 | No |
| gpt-5 | Naive vs Auto Conf Refine | 2.560 | 0.2192 | No |
| gpt-5 | Overthinking vs Auto Conf Refine | 0.190 | 1.0000 | No |

*p < 0.05 indicates statistically significant difference (marked with **)*
