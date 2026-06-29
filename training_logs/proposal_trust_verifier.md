# Proposal-Selector Trust Verifier

- target precision: `0.80`
- motif radius: `4`
- motif min similarity: `0.84`
- motif exclude radius: `16`

| System | Precision | Recall | F1 | Rescued rows | Replace Top-1 | Replace Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C2_motif_hgb | 0.8004 | 0.6123 | 0.6938 | 0 | 0.7918 | 0.9210 |
| proposal_trust_hgb | 0.8090 | 0.6001 | 0.6891 | 0 | 0.7919 | 0.9212 |
| C2_motif_proposal_trust_rescue | 0.8014 | 0.6110 | 0.6933 | 41 | 0.7911 | 0.9208 |
| C2_motif_narrow_trust_rescue | 0.8025 | 0.6101 | 0.6932 | 9 | 0.7916 | 0.9213 |
| C2_motif_proposal_trust_hgb | 0.8055 | 0.5956 | 0.6849 | 0 | 0.7946 | 0.9228 |
| C2_motif_full_trust_rescue | 0.8120 | 0.6001 | 0.6901 | 13 | 0.7949 | 0.9239 |
| C2_motif_narrow_full_rescue | 0.8022 | 0.6099 | 0.6929 | 8 | 0.7919 | 0.9217 |
