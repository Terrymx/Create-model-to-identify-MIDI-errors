# Multi-Window Surprise Validation

- checkpoint: `checkpoints\transformer_explicit_surprise_step2.pt`
- split: `validation`
- candidate threshold: `0.55`
- candidates: `14783`
- candidate precision: `0.6073`
- candidate recall ceiling: `0.6316`

| Signal | AUC | TP Mean | FP Mean | Mean Gap |
| --- | ---: | ---: | ---: | ---: |
| surprise_64 | 0.5364 | 8.7843 | 8.2633 | 0.5210 |
| surprise_128 | 0.5434 | 8.5416 | 7.8640 | 0.6776 |
| surprise_256 | 0.5426 | 7.5257 | 6.9351 | 0.5906 |
| surprise_mean | 0.5512 | 8.2838 | 7.6874 | 0.5964 |
| surprise_min | 0.5522 | 5.6418 | 5.0524 | 0.5894 |
| surprise_max | 0.5373 | 11.0073 | 10.4722 | 0.5351 |
| surprise_range | 0.4911 | 5.3655 | 5.4198 | -0.0543 |

- best signal: `surprise_min` (AUC `0.5522`)
- 256-window AUC: `0.5426`
- incremental AUC: `0.0096`
- recommendation: Do not retrain yet; the tested scales are mostly redundant, so inspect scale alignment or move to the cascade probe.
