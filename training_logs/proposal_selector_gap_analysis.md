# Proposal Selector Gap Analysis

## Oracle Detection Gap

- total errors: `15639`
- current true positives: `9575`
- candidate FN rows: `3326`
- candidate FN rows with target in proposals: `2199`
- ideal recall if all target-present candidate FN were rescued: `0.7529`
- ideal recall if all candidate FN rows were rescued: `0.8249`

## Candidate FN Groups

| Group | Count | Target present | Motif matched | Positive motif gain | Near threshold | Far below | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate_fn | 3326 | 0.6612 | 0.3274 | 0.2216 | 0.0734 | 0.7084 | 0.2574 |
| target_present_fn | 2199 | 1.0000 | 0.3597 | 0.2406 | 0.0814 | 0.6785 | 0.2725 |
| replace_target_present_fn | 2199 | 1.0000 | 0.3597 | 0.2406 | 0.0814 | 0.6785 | 0.2725 |

## Selector Target-Rank Accuracy

| Rows | Selector | Target present | Top-1 | Top-2 | Top-3 | Mean rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all_candidate_positives | B_ranking | 8574 | 0.4479 | 0.5514 | 0.6187 | 1.5654 |
| all_candidate_positives | C_aligned_ranking | 8574 | 0.4479 | 0.5514 | 0.6355 | 1.5402 |
| all_candidate_positives | E1_proposal_scores | 8574 | 0.5568 | 0.6372 | 0.6599 | 1.2105 |
| true_positives | B_ranking | 6375 | 0.4685 | 0.5636 | 0.6254 | 1.5106 |
| true_positives | C_aligned_ranking | 6375 | 0.4685 | 0.5636 | 0.6445 | 1.4819 |
| true_positives | E1_proposal_scores | 6375 | 0.5687 | 0.6409 | 0.6615 | 1.1896 |
| candidate_fn | B_ranking | 2199 | 0.3888 | 0.5162 | 0.5995 | 1.7244 |
| candidate_fn | C_aligned_ranking | 2199 | 0.3888 | 0.5162 | 0.6094 | 1.7094 |
| candidate_fn | E1_proposal_scores | 2199 | 0.5225 | 0.6263 | 0.6554 | 1.2710 |
| candidate_fn_target_present | B_ranking | 2199 | 0.5880 | 0.7808 | 0.9068 | 1.7244 |
| candidate_fn_target_present | C_aligned_ranking | 2199 | 0.5880 | 0.7808 | 0.9218 | 1.7094 |
| candidate_fn_target_present | E1_proposal_scores | 2199 | 0.7904 | 0.9472 | 0.9914 | 1.2710 |
| replace_fn_target_present | B_ranking | 2199 | 0.5880 | 0.7808 | 0.9068 | 1.7244 |
| replace_fn_target_present | C_aligned_ranking | 2199 | 0.5880 | 0.7808 | 0.9218 | 1.7094 |
| replace_fn_target_present | E1_proposal_scores | 2199 | 0.7904 | 0.9472 | 0.9914 | 1.2710 |
