# Candidate Verifier Probe

- base checkpoint: `checkpoints\transformer_explicit_surprise_step2.pt`
- Stage 1 threshold: `0.45`
- verifier threshold selected on calibration files: `0.41`
- test candidate recall ceiling: `0.7153`

| System | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 candidates | 0.5758 | 0.7153 | 0.6380 | 11784 | 8681 | 4690 |
| Stage 1 + verifier | 0.7980 | 0.5460 | 0.6484 | 8995 | 2277 | 7479 |

## Test Threshold Sweep

| Threshold | Precision | Recall | F1 | F0.5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.5804 | 0.7135 | 0.6401 | 0.6029 |
| 0.20 | 0.6465 | 0.6717 | 0.6588 | 0.6514 |
| 0.30 | 0.7365 | 0.6070 | 0.6655 | 0.7063 |
| 0.40 | 0.7933 | 0.5511 | 0.6504 | 0.7292 |
| 0.41 | 0.7980 | 0.5460 | 0.6484 | 0.7306 |
| 0.50 | 0.8351 | 0.5061 | 0.6303 | 0.7390 |
| 0.60 | 0.8667 | 0.4638 | 0.6042 | 0.7384 |
| 0.70 | 0.8988 | 0.4173 | 0.5699 | 0.7303 |
| 0.80 | 0.9297 | 0.3522 | 0.5109 | 0.7001 |
| 0.90 | 0.9638 | 0.2443 | 0.3898 | 0.6066 |
