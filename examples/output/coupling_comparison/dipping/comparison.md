# Six couplings, dipping: three dense bodies (+0.3 g/cc), χ = 0.05 / 0.01 / 0 SI

357 stations of gz and TMI (vertical field), 2% noise; L1–L2 (IRLS, L1 share 0.8) per model; unit-free coupling weight 1.

| coupling | A ρ / χ (0.3 / 0.05) | B ρ / χ (0.3 / 0.01) | C ρ / χ (0.3 / 0) | ρ rms error | χ rms error | χ² grav / mag | s | note |
|---|---|---|---|---|---|---|---|---|
| none (inverted together, uncoupled) | 0.045 / 0.0141 | 0.067 / 0.0008 | 0.055 / -0.0000 | 0.0441 | 0.00387 | 386 / 391 (N = 357) | 43 | one β for both models, no coupling: the reference |
| cross-gradient | 0.084 / 0.0141 | 0.099 / 0.0015 | 0.078 / -0.0000 | 0.0412 | 0.00387 | 383 / 385 (N = 357) | 110 |  |
| joint total variation | 0.047 / 0.0131 | 0.061 / 0.0008 | 0.052 / -0.0000 | 0.0434 | 0.00390 | 386 / 391 (N = 357) | 106 |  |
| linear correspondence | 0.104 / 0.0166 | 0.035 / 0.0030 | 0.021 / 0.0004 | 0.0427 | 0.00373 | 380 / 385 (N = 357) | 107 | ρ = 6 χ (the high-χ body's ratio) imposed everywhere |
| petrophysically guided (PGI) | 0.123 / 0.0209 | 0.105 / 0.0035 | 0.109 / 0.0001 | 0.0493 | 0.00501 | 186 / 339 (N = 357) | 30 | the true rock units given (full petrophysical information); PGI replaces L1–L2 |
| group lasso (joint sparsity) | 0.164 / 0.0301 | 0.135 / 0.0046 | 0.163 / 0.0001 | 0.0451 | 0.00484 | 387 / 438 (N = 357) | 222 | Utsugi (2025): L2 (λ2 = 0.3) + group lasso, λ1 at the L-curve corner; replaces L1–L2 |
| none — L1 + L2 by ADMM (control) | 0.153 / 0.0289 | 0.132 / 0.0021 | 0.162 / 0.0000 | 0.0438 | 0.00611 | 400 / 479 (N = 357) | 171 | control: the group lasso's solver and settings, each model soft-thresholded alone |
| group_lasso_balanced | 0.160 / 0.0297 | 0.131 / 0.0048 | 0.159 / 0.0002 | 0.0423 | 0.00458 | 335 / 385 (N = 357) | 24 | group lasso, paper's cell weights; errors as data scaling, chi^2 = N, datasets balanced |
| group_lasso_sv | 0.160 / 0.0297 | 0.131 / 0.0048 | 0.159 / 0.0002 | 0.0422 | 0.00455 | 333 / 387 (N = 357) | 14 | group lasso, paper's weights x cell volume; errors, chi^2 = N, balanced |
| group_lasso_depth | 0.147 / 0.0189 | 0.139 / 0.0037 | 0.159 / 0.0001 | 0.0480 | 0.00370 | 344 / 369 (N = 357) | 28 | group lasso, volume x depth weight (beta = 1); errors, chi^2 = N, balanced |

PGI cells in the right unit: {'A': '18/84', 'B': '19/84', 'C': '33/84'}; cells called a body outside the bodies: 367.
