# Proposal-Trust Rescue Audit

## System Delta

| System | Precision | Recall | TP delta | FP delta | Rescued |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2_motif_hgb | 0.8004 | 0.6123 | 0 | 0 | 0 |
| proposal_trust_hgb | 0.8090 | 0.6001 | -190 | -172 | 0 |
| C2_motif_proposal_trust_rescue | 0.8014 | 0.6110 | -20 | -20 | 41 |
| C2_motif_narrow_trust_rescue | 0.8025 | 0.6101 | -33 | -39 | 9 |
| C2_motif_proposal_trust_hgb | 0.8055 | 0.5956 | -260 | -139 | 0 |
| C2_motif_full_trust_rescue | 0.8120 | 0.6001 | -190 | -215 | 13 |
| C2_motif_narrow_full_rescue | 0.8022 | 0.6099 | -37 | -36 | 8 |

## Group Summary

| Group | Count | Positive rate | Target present | Mean C2 | Mean rescue |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_candidate_fn | 3326 | 1.0000 | 0.6612 | 0.2574 | 0.1332 |
| baseline_false_positive | 2388 | 0.0000 | 0.0000 | 0.7752 | 0.9439 |
| narrow_trust_eligible | 4338 | 0.1625 | 0.1272 | 0.1807 | 0.0758 |
| narrow_trust_eligible_tp | 705 | 1.0000 | 0.7830 | 0.2928 | 0.1656 |
| narrow_trust_eligible_fp | 3633 | 0.0000 | 0.0000 | 0.1589 | 0.0584 |
| narrow_trust_rescued | 9 | 0.6667 | 0.2222 | 0.5562 | 0.9617 |
| narrow_full_eligible | 5151 | 0.1275 | 0.0938 | 0.1288 | 0.0227 |
| narrow_full_rescued | 8 | 0.2500 | 0.0000 | 0.3487 | 0.8753 |

## Top-K Eligible by Rescue Score

| Group | Count | Positive rate | Target present | E1 Top-1 | E1 Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| narrow_trust_eligible_rejected_top25 | 25 | 0.5200 | 0.2000 | 1.0000 | 1.0000 |
| narrow_trust_eligible_rejected_top50 | 50 | 0.4600 | 0.2400 | 1.0000 | 1.0000 |
| narrow_trust_eligible_rejected_top100 | 100 | 0.4000 | 0.2400 | 1.0000 | 1.0000 |
| narrow_trust_eligible_rejected_top200 | 200 | 0.3700 | 0.2600 | 1.0000 | 1.0000 |
| narrow_trust_eligible_rejected_top500 | 500 | 0.3580 | 0.2680 | 0.9851 | 1.0000 |
| narrow_full_eligible_rejected_top25 | 25 | 0.3600 | 0.2400 | 0.8333 | 1.0000 |
| narrow_full_eligible_rejected_top50 | 50 | 0.4600 | 0.3000 | 0.8000 | 1.0000 |
| narrow_full_eligible_rejected_top100 | 100 | 0.4300 | 0.3100 | 0.8710 | 1.0000 |
| narrow_full_eligible_rejected_top200 | 200 | 0.3350 | 0.2500 | 0.9000 | 1.0000 |
| narrow_full_eligible_rejected_top500 | 500 | 0.2660 | 0.1920 | 0.9271 | 1.0000 |

## Largest Trust-Feature Deltas: Narrow Eligible TP vs FP

| Feature | TP mean | FP mean | Delta |
| --- | ---: | ---: | ---: |
| e1_best | 1.9426 | 1.2620 | 0.6806 |
| e1_margin | 4.9767 | 4.7083 | 0.2684 |
| b_at_e1_best | 2.9224 | 2.6889 | 0.2335 |
| c_at_e1_best | 2.9224 | 2.6889 | 0.2335 |
| c2_score | 0.2928 | 0.1589 | 0.1338 |
| e1_entropy | 0.1636 | 0.1777 | -0.0141 |
| best_pitch | 0.5061 | 0.5126 | -0.0066 |
| observed_pitch | 0.5062 | 0.5126 | -0.0064 |
| e1_best_prob | 0.9598 | 0.9558 | 0.0040 |
| e1_c_agree | 0.9929 | 0.9895 | 0.0034 |
| b_c_agree | 0.9929 | 0.9895 | 0.0034 |
| best_delta | -0.0009 | -0.0000 | -0.0009 |
