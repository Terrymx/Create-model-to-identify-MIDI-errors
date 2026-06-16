# Union Candidate Verifier Variants

- target precision: `0.8`
- test candidate recall ceiling: `0.7297`
- test candidate precision: `0.5121`

| Variant | Best epoch | Cal threshold | Cal P | Cal R | Test P | Test R | Test frontier P | Test frontier R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_balanced | 67 | 0.57 | 0.8022 | 0.4988 | 0.7971 | 0.5650 | 0.8004 | 0.5614 |
| linear_precision | 60 | 0.50 | 0.8001 | 0.4991 | 0.7966 | 0.5684 | 0.8007 | 0.5648 |
| linear_high_precision | 42 | 0.47 | 0.8004 | 0.5002 | 0.7956 | 0.5689 | 0.8002 | 0.5649 |
| shallow_precision | 32 | 0.34 | 0.8003 | 0.5081 | 0.7957 | 0.5731 | 0.8042 | 0.5650 |
| mlp_balanced | 9 | 0.49 | 0.8016 | 0.5054 | 0.7967 | 0.5708 | 0.8008 | 0.5664 |
| mlp_precision | 9 | 0.35 | 0.8002 | 0.5022 | 0.8013 | 0.5673 | 0.8013 | 0.5673 |
| mlp_high_precision | 23 | 0.20 | 0.8001 | 0.5102 | 0.8006 | 0.5708 | 0.8006 | 0.5708 |
| sklearn_logreg_plain | - | 0.50 | 0.8037 | 0.5050 | 0.8043 | 0.5694 | 0.8043 | 0.5694 |
| sklearn_logreg_balanced | - | 0.51 | 0.8051 | 0.5022 | 0.8049 | 0.5667 | 0.8014 | 0.5723 |
| sklearn_linear_svm_balanced | - | 0.50 | 0.8008 | 0.5040 | 0.8028 | 0.5693 | 0.8028 | 0.5693 |
| sklearn_rbf_svm_balanced | - | 0.46 | 0.8025 | 0.4729 | 0.8034 | 0.5384 | 0.8034 | 0.5384 |
| sklearn_random_forest_balanced | - | 0.50 | 0.8020 | 0.4995 | 0.8050 | 0.5657 | 0.8050 | 0.5657 |
| sklearn_hist_gradient_boosting | - | 0.51 | 0.8026 | 0.5140 | 0.8031 | 0.5762 | 0.8031 | 0.5762 |
