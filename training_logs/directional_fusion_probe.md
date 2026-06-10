# Directional Fusion Probe

- candidate threshold: `0.45`
- target precision: `0.8`
- fusion epoch selected on calibration files: `10`
- test candidate recall ceiling: `0.7176`

| System | Selected on | Threshold | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Step 2 | calibration | 0.85 | 0.8023 | 0.5328 | 0.6404 |
| Directional fusion | calibration | 0.41 | 0.7927 | 0.5752 | 0.6667 |
| Directional fusion | test diagnostic | 0.43 | 0.8024 | 0.5658 | 0.6636 |

- test-frontier recall gain over historical Step 2 `0.5307`: `+0.0351`
- probe passed +0.03 recall Gate: `True`
