# Directional Likelihood Signal Gate

- split: `validation`
- error rate: `0.01`
- valid notes with both directions: `1251966`
- error notes: `14102`
- candidate threshold: `0.55`
- candidate precision: `0.6082`
- candidate recall ceiling: `0.6337`

| Population | Signal | AUC | Clean/FP mean | Error/TP mean | JS divergence |
| --- | --- | ---: | ---: | ---: | ---: |
| all notes | forward | 0.8495 | 2.4919 | 5.0127 | 0.2148 |
| all notes | backward | 0.8618 | 2.3773 | 5.2337 | 0.2316 |
| all notes | directional_mean | 0.9101 | 2.4346 | 5.1232 | 0.3145 |
| all notes | directional_min | 0.9062 | 1.5531 | 4.2739 | 0.3056 |
| all notes | directional_max | 0.8683 | 3.3161 | 5.9724 | 0.2428 |
| all notes | directional_disagreement | 0.4893 | 1.7630 | 1.6986 | 0.0011 |
| all notes | step2_internal | 0.7868 | 2.8231 | 6.6166 | 0.1350 |
| candidates | forward | 0.6162 | 4.7365 | 5.4772 | 0.0264 |
| candidates | backward | 0.6382 | 4.8123 | 5.7847 | 0.0334 |
| candidates | directional_mean | 0.6633 | 4.7744 | 5.6310 | 0.0455 |
| candidates | directional_min | 0.6562 | 3.8599 | 4.7638 | 0.0435 |
| candidates | directional_max | 0.6370 | 5.6888 | 6.4982 | 0.0330 |
| candidates | directional_disagreement | 0.4838 | 1.8289 | 1.7344 | 0.0037 |
| candidates | step2_internal | 0.5428 | 6.9382 | 7.5324 | 0.0062 |

- Step 2 candidate AUC: `0.5428`
- best directional candidate signal: `directional_mean`
- best directional candidate AUC: `0.6633`
- Gate target AUC: `0.5628`
- Gate passed: `True`
- recommendation: The directional evidence passes the hard-candidate separation Gate. Proceed to a frozen-Step-2 fusion probe before full detector retraining.
