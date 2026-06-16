# Traditional End-to-End Baseline

- train error rate: `0.08`
- eval error rate: `0.01`
- target precision: `0.8`
- train notes kept: `500000` / seen `11193985`
- validation notes: `1261824`
- test notes: `1457920`

| Model | Cal threshold | Cal P | Cal R | Test P | Test R | Test frontier P | Test frontier R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression_plain | 0.71 | 0.3500 | 0.0005 | 0.2800 | 0.0004 | 1.0000 | 0.0001 |
| logistic_regression_balanced | 0.96 | 0.2778 | 0.0004 | 0.2308 | 0.0004 | 0.2500 | 0.0001 |
| linear_svm_balanced | 0.67 | 0.2800 | 0.0005 | 0.2121 | 0.0004 | 1.0000 | 0.0001 |
| random_forest_balanced | 0.82 | 0.8298 | 0.0198 | 0.8836 | 0.0242 | 0.8056 | 0.0345 |
| hist_gradient_boosting | 0.96 | 0.8105 | 0.0872 | 0.8523 | 0.0924 | 0.8523 | 0.0924 |
