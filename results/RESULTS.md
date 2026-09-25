# CIFAR-10 test results

Mean and sample standard deviation over downstream training seeds 41, 42, and 43. All runs use the same seed-42 SimCLR encoder and the same seed-42 data split.

| Labels | Method | Seed 41 | Seed 42 | Seed 43 | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---|---:|---:|---:|---:|---:|
| 1% | Scratch | 40.88% | 41.28% | 40.72% | **40.96 ± 0.29%** | **40.60 ± 0.51%** |
| 1% | Linear Probe | 67.78% | 67.59% | 67.76% | **67.71 ± 0.10%** | **67.47 ± 0.17%** |
| 1% | Fine-tune | 67.50% | 68.30% | 67.25% | **67.68 ± 0.55%** | **67.48 ± 0.71%** |
| 10% | Scratch | 71.19% | 72.79% | 70.86% | **71.61 ± 1.03%** | **71.54 ± 1.10%** |
| 10% | Linear Probe | 73.43% | 73.32% | 73.42% | **73.39 ± 0.06%** | **73.10 ± 0.06%** |
| 10% | Fine-tune | 79.67% | 79.75% | 79.82% | **79.75 ± 0.08%** | **79.70 ± 0.08%** |
