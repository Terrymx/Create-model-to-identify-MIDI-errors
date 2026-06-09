# External Likelihood Diagnostic

- split: `validation`
- error rate: `0.01`
- valid notes: `1261824`
- error notes: `14214`
- legacy teacher clean perplexity: `1294.5382`
- train-aligned teacher clean perplexity: `8.0609`
- teacher/base Pearson correlation: `0.3226`
- teacher/base Spearman correlation: `0.3242`

| Population | Signal | AUC | Clean/FP Mean | Error/TP Mean | JS Divergence |
| --- | --- | ---: | ---: | ---: | ---: |
| all notes | external_teacher | 0.5860 | 7.1659 | 8.6368 | 0.0120 |
| all notes | external_teacher_train_aligned | 0.6284 | 2.0870 | 3.3840 | 0.0254 |
| all notes | step2_internal | 0.7862 | 2.8277 | 6.6094 | 0.1345 |
| candidates | external_teacher | 0.4977 | 8.8939 | 8.8496 | 0.0031 |
| candidates | external_teacher_train_aligned | 0.5302 | 3.3917 | 3.7673 | 0.0057 |
| candidates | step2_internal | 0.5426 | 6.9351 | 7.5257 | 0.0062 |

- candidate threshold: `0.55`
- candidate precision: `0.6073`
- candidate recall ceiling: `0.6316`
- recommendation: The teacher signal is too weak among hard candidates. Improve the clean-music model or its masking objective before another detector training run.
