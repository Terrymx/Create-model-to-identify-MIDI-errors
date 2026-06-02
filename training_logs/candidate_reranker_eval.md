# Candidate Reranker Evaluation

- checkpoint: `checkpoints\transformer_melodic_theory_precision.pt`
- candidate threshold: `0.85`
- train candidates: `97347`
- eval candidates: `15040`

| Rerank Threshold | Precision | Recall | F1 | F0.5 | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.7841 | 0.9316 | 0.8515 | 0.8098 | 10126 | 2788 | 743 |
| 0.40 | 0.8169 | 0.8665 | 0.8410 | 0.8264 | 9418 | 2111 | 1451 |
| 0.50 | 0.8489 | 0.7735 | 0.8095 | 0.8327 | 8407 | 1496 | 2462 |
| 0.60 | 0.8887 | 0.6254 | 0.7342 | 0.8197 | 6798 | 851 | 4071 |
| 0.70 | 0.9302 | 0.3972 | 0.5567 | 0.7334 | 4317 | 324 | 6552 |
