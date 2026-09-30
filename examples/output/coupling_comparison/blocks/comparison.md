# Six couplings, blocks: three dense bodies (+0.3 g/cc), χ = 0.05 / 0.01 / 0 SI

357 stations of gz and TMI (vertical field), 2% noise; L1–L2 (IRLS, L1 share 0.8) per model; unit-free coupling weight 1.

| coupling | A ρ / χ (0.3 / 0.05) | B ρ / χ (0.3 / 0.01) | C ρ / χ (0.3 / 0) | ρ rms error | χ rms error | χ² grav / mag | s | note |
|---|---|---|---|---|---|---|---|---|
| none (inverted together, uncoupled) | 0.079 / 0.0224 | 0.105 / 0.0011 | 0.080 / -0.0000 | 0.0423 | 0.00342 | 379 / 385 (N = 357) | 39 | one β for both models, no coupling: the reference |
| cross-gradient | 0.095 / 0.0187 | 0.153 / 0.0023 | 0.115 / 0.0000 | 0.0380 | 0.00356 | 394 / 399 (N = 357) | 80 |  |
| joint total variation | 0.074 / 0.0201 | 0.088 / 0.0011 | 0.072 / -0.0000 | 0.0421 | 0.00350 | 378 / 385 (N = 357) | 66 |  |
| linear correspondence | 0.164 / 0.0266 | 0.048 / 0.0040 | 0.029 / 0.0005 | 0.0420 | 0.00312 | 384 / 390 (N = 357) | 64 | ρ = 6 χ (the high-χ body's ratio) imposed everywhere |
| petrophysically guided (PGI) | 0.127 / 0.0364 | 0.188 / 0.0067 | 0.215 / 0.0002 | 0.0436 | 0.00423 | 131 / 160 (N = 357) | 40 | the true rock units given (full petrophysical information); PGI replaces L1–L2 |
| group lasso (joint sparsity) | 0.267 / 0.0488 | 0.245 / 0.0088 | 0.252 / 0.0002 | 0.0342 | 0.00438 | 382 / 436 (N = 357) | 181 | Utsugi (2025): L2 (λ2 = 0.3) + group lasso, λ1 at the L-curve corner; replaces L1–L2 |
| none — L1 + L2 by ADMM (control) | 0.247 / 0.0493 | 0.233 / 0.0092 | 0.251 / 0.0000 | 0.0331 | 0.00604 | 401 / 484 (N = 357) | 213 | control: the group lasso's solver and settings, each model soft-thresholded alone |

PGI cells in the right unit: {'A': '48/96', 'B': '40/96', 'C': '70/96'}; cells called a body outside the bodies: 384.
