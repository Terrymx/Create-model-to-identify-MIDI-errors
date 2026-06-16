# Union Candidate Verifier

- threeclass candidate threshold: `0.45`
- binary candidate threshold: `0.45`
- target precision: `0.8`
- verifier epoch selected on calibration files: `13`
- test candidate recall ceiling: `0.7297`
- test candidate precision: `0.5121`

| System | Selected on | Threshold | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| threeclass | test frontier | 0.69 | 0.8038 | 0.5276 | 0.6371 |
| binary | test frontier | 0.85 | 0.8039 | 0.4929 | 0.6111 |
| max | test frontier | 0.87 | 0.8044 | 0.5187 | 0.6307 |
| verifier | calibration | 0.38 | 0.7508 | 0.6131 | 0.6750 |
| verifier | test frontier | 0.50 | 0.8036 | 0.5623 | 0.6616 |
