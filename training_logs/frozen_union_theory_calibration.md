# Union Candidate Context Calibration

- target precision: `0.8`

| Run | Strategy | Best selected test P | Best selected test R | Selected cal P | Selected cal R |
| --- | --- | ---: | ---: | ---: | ---: |
| three=0.60,binary=0.50/hist_gradient_boosting | raw | 0.8082 | 0.5844 | 0.8229 | 0.5088 |
| three=0.60,binary=0.50/hist_gradient_boosting | density_adjust_alpha=0.02 | 0.8022 | 0.5916 | 0.8133 | 0.5188 |
| three=0.60,binary=0.50/hist_gradient_boosting | density_adjust_alpha=0.04 | 0.8029 | 0.5907 | 0.8108 | 0.5188 |
| three=0.60,binary=0.50/hist_gradient_boosting | density_adjust_alpha=0.06 | 0.8001 | 0.5932 | 0.8001 | 0.5261 |
| three=0.60,binary=0.50/hist_gradient_boosting | density_adjust_alpha=0.08 | 0.8056 | 0.5884 | 0.8019 | 0.5242 |
| three=0.60,binary=0.50/hist_gradient_boosting | density_adjust_alpha=0.10 | 0.8081 | 0.5839 | 0.8002 | 0.5219 |
| three=0.60,binary=0.50/hist_gradient_boosting | rank_blend_beta=0.10 | 0.8010 | 0.5915 | 0.8119 | 0.5158 |
| three=0.60,binary=0.50/hist_gradient_boosting | rank_blend_beta=0.20 | 0.8041 | 0.5884 | 0.8136 | 0.5150 |
| three=0.60,binary=0.50/hist_gradient_boosting | rank_blend_beta=0.30 | 0.8022 | 0.5894 | 0.8102 | 0.5169 |
| three=0.60,binary=0.50/hist_gradient_boosting | rank_blend_beta=0.40 | 0.8039 | 0.5857 | 0.8108 | 0.5154 |
| three=0.60,binary=0.50/hist_gradient_boosting | piece_z_blend_gamma=0.02 | 0.8084 | 0.5844 | 0.8235 | 0.5092 |
| three=0.60,binary=0.50/hist_gradient_boosting | piece_z_blend_gamma=0.04 | 0.8080 | 0.5846 | 0.8217 | 0.5100 |
| three=0.60,binary=0.50/hist_gradient_boosting | piece_z_blend_gamma=0.06 | 0.8091 | 0.5834 | 0.8215 | 0.5092 |
| three=0.60,binary=0.50/hist_gradient_boosting | piece_z_blend_gamma=0.08 | 0.8004 | 0.5923 | 0.8112 | 0.5169 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | raw | 0.8010 | 0.5918 | 0.8210 | 0.5165 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | density_adjust_alpha=0.02 | 0.8069 | 0.5881 | 0.8204 | 0.5127 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | density_adjust_alpha=0.04 | 0.8046 | 0.5893 | 0.8120 | 0.5181 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | density_adjust_alpha=0.06 | 0.8056 | 0.5886 | 0.8119 | 0.5208 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | density_adjust_alpha=0.08 | 0.8013 | 0.5922 | 0.8001 | 0.5261 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | density_adjust_alpha=0.10 | 0.8102 | 0.5790 | 0.8008 | 0.5161 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | rank_blend_beta=0.10 | 0.8044 | 0.5884 | 0.8241 | 0.5150 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | rank_blend_beta=0.20 | 0.8056 | 0.5867 | 0.8229 | 0.5142 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | rank_blend_beta=0.30 | 0.8009 | 0.5902 | 0.8120 | 0.5181 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | rank_blend_beta=0.40 | 0.8024 | 0.5870 | 0.8108 | 0.5173 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | piece_z_blend_gamma=0.02 | 0.8027 | 0.5903 | 0.8223 | 0.5158 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | piece_z_blend_gamma=0.04 | 0.8044 | 0.5892 | 0.8237 | 0.5154 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | piece_z_blend_gamma=0.06 | 0.8045 | 0.5890 | 0.8241 | 0.5150 |
| three=0.60,binary=0.50/hist_gradient_boosting_small_leaf | piece_z_blend_gamma=0.08 | 0.8032 | 0.5905 | 0.8205 | 0.5165 |
