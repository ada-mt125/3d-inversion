# Joint gravity–magnetic inversion with L2 + group lasso (Utsugi 2025)

The group lasso is one of the joint inversion's couplings (`coupling="group_lasso"`;
`regularization_type="group_lasso"` still means it): see `docs/joint_couplings.md` for
the others. Its optional cross-gradient (`gl_cross_gradient`, λ3) is our extension, not
the paper's method: such runs are recorded as the coupling `group_lasso+cross_gradient`
("hybrid").

Code: `geoinv3d/methods/group_lasso.py` (solver), `geoinv3d/cloud/worker.py`
(`run_group_lasso_joint`, the data pipeline), tests: `tests/test_group_lasso.py`.

Reference: M. Utsugi (2025), *Joint inversion of magnetic and gravity data using
group lasso regularization*, Earth, Planets and Space 77, 146,
https://doi.org/10.1186/s40623-025-02270-1.

> The paper's full text could not be fetched while this was written (the
> publisher's site needs a sign-in). "Follows the paper" below means: follows the
> specification of the paper's method that we were given. Numbers that belong to
> the paper (its λ values, its synthetic model) were not used or checked.

## The problem

With the magnetic model β and the density model ρ on the same M active cells,
f = Xβ, g = Yρ, ζ = [β, ρ], b = [f, g] and Z = diag(X, Y):

    minimize  ½‖b − Zζ‖² + λ1 Σₖ ‖(βₖ, ρₖ)‖₂ + ½λ2 ‖ζ‖²

The group of cell k is its two properties. The group norm √(βₖ² + ρₖ²) either
empties a cell in both models or selects it for both; a selected cell may keep
one component at zero, and the two need not share a sign. λ2 is plain amplitude
damping (no spatial derivatives): it spreads the solution that the group lasso
alone concentrates into a few cells. This is not the same as two separate
sparse inversions, nor as λ(‖β‖₁ + ‖ρ‖₁): those have no coupling at all
(`coupling="none"` gives the latter as a reference; Test A below compares them).

## What follows the paper

| Part | Implementation |
|---|---|
| Objective: L2 misfit + λ1 × group lasso over (βₖ, ρₖ) + ½λ2‖ζ‖² | `JointGroupLassoProblem.solve` |
| Sensitivity weighting wⱼ = (1/‖Kⱼ‖)^(γ/2), γ = 2, X = K W_K, Y = G W_G; physical model = W × scaled model | `sensitivity_weights` |
| Data balance C = max\|f\| / max\|g\|, S_f = I, S_g = C I (a configurable step, not hard-coded) | `data_scaling(mode="max_ratio")`, the default |
| ADMM with s = ζ and the scaled dual u: ζ-update, s-update, u ← u + s − ζ | `solve` |
| ζ-update as two independent systems (XᵀX + μI)β = Xᵀf + μ(s_β + u_β), same for ρ | `_DataSpaceCholesky`, `_ModelSpaceCholesky`, `_ConjugateGradient` |
| s-update: group soft threshold sₖ = μ/(μ + λ2) · max(1 − λ1/(μ‖qₖ‖), 0) · qₖ, q = ζ − u | `group_shrink` |
| λ2 fixed first, λ1 from the L-curve of log10(misfit) against log10(group penalty) | `lcurve`, `plot_lcurve` |
| Bounds (e.g. susceptibility ≥ 0) — an addition, not in the paper | `bounds`, `_group_shrink_box` (below) |

## Engineering choices (not from the paper)

**Where the data scale goes.** A scalar scale C multiplies the gravity data only;
the operator keeps its unit columns and the scaled density variable absorbs C
(physical ρ = w · ρ̃ / C). Scaling the operator's rows by C as well would leave
the variable in mGal-like units while β̃ is in nT-like units, so the group norm
would weigh the two components hundreds of times differently (our first version
did this; ADMM then left the model at zero). A per-datum scale cannot be absorbed
by the model: `data_scaling="std"` (each datum ÷ σ, misfit = χ²) weights the
operator's rows and computes the column weights from the weighted rows.

**Large problems.**

- X, Y and Z are never formed: `WeightedOperator` applies diag(S) K diag(w) on
  the fly, and float32 sensitivities (as SimPEG stores them) stay float32
  (vectors are cast to float32 for the products, not the matrix to float64).
- No M × M or 2M × 2M matrix when there are fewer data than cells: by Woodbury,
  (XᵀX + μI)⁻¹ = (I − Xᵀ(XXᵀ + μI)⁻¹X)/μ, so the ζ-update needs one Cholesky
  factor of the N × N matrix XXᵀ + μI per method. It is made once (a chunked
  symmetric rank-k update, no N × M float64 copy) and reused for every ADMM
  iteration and every λ1 of a sweep, because X, Y and μ do not change. Each
  iteration then costs two products with each sensitivity (three with history
  recording). With fewer cells than data the M × M factor is used instead.
- When a factor would exceed 4 GB (`factor_max_bytes`) or the operator is
  matrix-free (a `LinearOperator`, e.g. SimPEG's Jvec/Jtvec), the systems are
  solved by conjugate gradients, warm-started from the previous ζ.
- The s-update is one vectorized O(M) pass; no loop over cells.
- float32 products leave the ADMM residuals a floor of about 10⁻⁶ (relative):
  fine for the default tolerance of 10⁻⁴, not for 10⁻⁷ (use float64 kernels).

**Stopping.** Boyd et al. (2011, §3.3.1): primal ‖s − ζ‖ ≤ √(2M)·ε_abs + ε_rel·max(‖ζ‖, ‖s‖)
and dual μ‖s − s_old‖ ≤ √(2M)·ε_abs + ε_rel·‖μu‖, with ε_rel = 10⁻⁴
(`tol_primal`, `tol_dual`) and ε_abs = 10⁻⁹ max|Zᵀb| by default, or `max_iter`
(then `converged` is False). Never on the objective alone.

**Defaults and helpers.**

- λ1,max = maxₖ ‖(Zᵀb)ₖ‖ is where the solution becomes exactly zero (closed
  form, no iterations); the sweep runs from just below it down `decades`
  decades with warm starts.
- μ defaults to the mean non-zero eigenvalue of XᵀX (M / min(N, M) with unit
  columns); on the synthetic, ADMM was fastest near it (μ = 10 vs the default 14;
  0.1 and 100 were several times slower). μ changes the path, not the minimizer
  (tested).
- λ2 defaults to 0.3 — our choice from the synthetic below, not the paper's.
- The returned model is the s iterate (exact zeros), not ζ.
- Pipeline: λ1 by the L-curve corner (maximum curvature of cubic splines in
  log λ1, the same code as the L1–L2 path), falling back to χ² = N when there is
  no corner, else the point nearest χ² = N; or χ² = N; or a fixed fraction of
  λ1,max. A warning names λ2 when χ² stays above N down to the smallest λ1.
- "Stop & keep result" on the Jobs page ends the sweep (λ1 is chosen from the
  points done) or the ADMM solve (its current iterate is kept).
- Results carry the observed and predicted data of both datasets
  (`data_gravity.npz`, `data_magnetics.npz`); the viewer shows a joint run as a
  density model and a susceptibility model, each with its own data fit.
- The magnetic model is SimPEG's susceptibility (SI), not magnetization (A/m);
  with γ = 2 the scaled problem does not depend on that choice of units.

## Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `lambda1` | group sparsity and structural coupling | L-curve on a sweep (pipeline) |
| `lambda2` | L2 damping of the unit-column variables (XᵀX has a unit diagonal) | 0.3 |
| `mu` | ADMM penalty: speed only, not a regularization parameter | mean eigenvalue of XᵀX |
| `gamma` | sensitivity weighting exponent | 2 |
| `cell_weights` | per model, each cell's weight R in the penalty (w = 1/R, scaled to unit columns on average) instead of `gamma` | none (worker `gl_weighting="depth"`: volume × depth weight) |
| `relaxation` | ADMM over-relaxation α | 1 (`GroupLassoProblem`), 1.6 (worker) |
| `data_scaling` | `"max_ratio"` (paper), `"std"`, `"none"`, `"auto"`, or scales | `"max_ratio"` (`JointGroupLassoProblem`), `"auto"` (worker) |
| `tol_primal`, `tol_dual`, `tol_abs` | ADMM stopping | 1e-4, 1e-4, 1e-9 max\|Zᵀb\| |
| `max_iter` | ADMM iterations per λ1 | 2000 (pipeline: 3000) |

Pipeline keys (`InversionTask`, `params_json`): `gl_lambda1_selection`
(`lcurve`/`discrepancy`/`fixed`), `gl_lambda1`, `gl_lambda1_ratio`, `gl_lambda2`,
`gl_mu`, `gl_data_scaling`, `gl_gamma`, `gl_n_lambda1` (13), `gl_lambda1_decades`
(3), `gl_max_iter`, `gl_tol`, `gl_cross_gradient`, `gl_gn_max_iter`, `gl_gn_tol`,
`gl_weighting` (`sensitivity`/`sensitivity_volume`/`depth`), `gl_relaxation` (1.6), `gl_balance` (False),
`gl_data_weights` (None), `gl_lambda1_selection="search"` (with `gl_lambda1_ratio` as the start),
`gl_balance_rounds` (6).  The pipeline's auto mode uses `GROUP_LASSO_AUTO`: `depth`
weighting with β = 1, `std` data scaling, λ1 for χ² = N, and the balance.

## Validation (tests/test_group_lasso.py)

Synthetic: 16 × 16 × 8 cells of 50 m under 12 × 12 stations, SimPEG TMI
(50 000 nT, I = 60°, D = 10°) and gz sensitivities, a 4 × 4 × 3-cell block
(0.05 SI, 0.3 g/cc), 2 % noise.

| Check | Result |
|---|---|
| Prox optimality (analytic certificate and a numerical minimum) | exact |
| KKT conditions at convergence (tolerance 10⁻⁷, float64) | < 10⁻⁴ of λ1 |
| Independent accelerated proximal gradient (FISTA) | same minimizer within 10⁻³; ADMM objective no higher |
| μ = 14 vs μ = 2.8 | same model within 10⁻⁴ |
| Cholesky (data / model space), CG, matrix-free LinearOperator | agree within 10⁻⁵ |
| float32 kernels at the default tolerance vs float64 | within 2 × 10⁻³ |
| A: coincident body, λ1 at the L-curve corner (0.022 λ1,max, λ2 = 1) | 86 % of \|β\| and 88 % of \|ρ\| within one cell of the body; overlap of the cells above 20 % of each model's peak (Jaccard) 0.68, vs 0.37 with separate soft thresholds at the same λ |
| B: magnetic-only body next to a coincident one (0.02 λ1,max, λ2 = 1) | mean \|β\| in it 1.2 × that in the coincident body, mean \|ρ\| 3 %; with no gravity signal at all, ρ is exactly 0 |
| C: gravity-only body, same settings | mean \|ρ\| in it 0.85 ×, mean \|β\| 2 % |
| D: β > 0 with ρ < 0, same settings | all 70 core cells have β > 0 and ρ < 0; correlation −0.97 |
| E: 0.01 λ1,max; cells holding 90 % of Σ‖sₖ‖² (peak ρ, true 0.3; χ², N = 288) | group lasso alone 6 (3.06; 291) · λ2 = 0.1: 39 (0.43; 286) · λ2 = 1: 89 (0.17; 342) · L2 alone, λ2 = 1: 447 (0.10; 79) |
| Pipeline: CSV files → worker → result.zip → viewer workflow | two models with their data fits; progress per λ1; stop & keep |

The test assertions are looser than these numbers (e.g. Jaccard > 0.5 and at
least 0.1 above the separate one; the other model below 15 % in B and C).

λ2 on the same synthetic (λ1 at χ² = N, 2 % noise):

| λ2 | peak ρ (true 0.3) | share of \|ρ\| in the body | ADMM iterations per λ1 |
|---|---|---|---|
| 0.01 | 1.36 | 1.00 | up to the 2000 limit |
| 0.1 | 0.45 | 0.81 | 130–810 |
| 0.3 | 0.28 | 0.59 | 130–300 |
| 1 | 0.16 | 0.37 | 70–240 (χ² floor 107 of N = 288) |

So λ2 trades compactness (and inflated values) against spread and fit; it is
data-dependent, and the paper's values should not be assumed to transfer.
The page offers a λ2 sweep (one job per value) for this.

## Cost

Memory: the two float32 sensitivities (4 bytes per entry, ×1.5 while SimPEG
builds them), plus min(N, M)² × 8 bytes per method for the Cholesky factors
(none with CG), plus the warm-start states of the sweep (4M floats per λ1).
The upload page's Review step estimates this against the instance.
Time: two products with each sensitivity per ADMM iteration (Cholesky path);
on the synthetic, 70–350 iterations per λ1 with warm starts.

## Beyond the paper: more models, nonlinear data, a cross-gradient

`GroupLassoProblem` generalizes the problem; `JointGroupLassoProblem` is its
two-model case and gives the same iterates as before, bit for bit.

- **Models and datasets.** P models, each explained by one or more datasets
  (`GroupLassoData`, with the index of its model); the group of cell k is
  (ζ₁ₖ, …, ζ_Pₖ).  A model's datasets are stacked row-wise (`StackedOperator`),
  its column weights come from all of them, and a scalar scale of its first
  dataset is carried by the model variable as above.  In the worker, datasets
  with the same model label (`joint_models`) share a model, e.g. gz and gzz of
  one density model.
- **Reference models.** m_p = ref_p + w_p ζ_p / c_p; potential fields use 0,
  MT and DC log(σ_background), so an "empty" cell is the background.
- **Nonlinear data (MT, DC).** Levenberg–Marquardt Gauss–Newton around ADMM:
  at ζ_k each dataset is linearized, b = s(d − F(m_k)) + X ζ_k with the
  simulation's J; ADMM solves the linearized problem plus ν/2‖ζ − ζ_k‖² on the
  nonlinear models only (their ζ systems get μ + ν; the linear models keep
  their factors); the step is accepted when the true objective decreases, and
  ν follows the ratio of actual to predicted decrease.  Stop: KKT residual
  < `gn_kkt_tol` (1e-2 of λ1), or an accepted step with ν ≤ μ that changes the
  objective by < `gn_tol` (1e-5), or `gn_max_iter` (20).  A backtracking line
  search on the undamped step stalled near the noise level (steps of ¼, the
  objective changing 1e-5 per step).  `data_scaling="auto"` uses 1/σ as soon as
  a dataset is nonlinear.
- **Cross-gradient.** λ3 C(ζ), C = Σ_{i<j} f_ij φ(u_i, u_j) with φ SimPEG's
  discretization of ∫|∇u_i × ∇u_j|² (`CrossGradientTerm`) and u_p = (w_p /
  median w_p) ζ_p the anomaly without the depth weighting.  f_ij sets the
  term's curvature, at the damped least-squares models (XᵀX + μI)⁻¹Xᵀb, equal
  to the data term's (largest eigenvalues by power iteration), so λ3 does not
  depend on the units of data or models and λ3 = 1 weighs the coupling like
  the data; 0.01–1 is the useful range on the synthetics.  For fixed other
  models C is quadratic and positive semidefinite in ζ_p, so the ζ update
  solves one SPD system per model, Gauss–Seidel with the latest other models,
  by CG preconditioned with the Cholesky solve (tolerance following the primal
  residual).  Fixed points satisfy the KKT conditions of the whole nonconvex
  objective (tested to 1e-3 at tight tolerances; without the coupling term's
  gradient they would be violated by > 1e-2).

| Parameter | Meaning | Default |
|---|---|---|
| `gl_cross_gradient` | λ3, unit-free weight of the cross-gradient | 0 (off) |
| `gl_gn_max_iter` | Gauss–Newton linearizations per λ1 (nonlinear data) | 20 |
| `gl_gn_tol` | relative objective change that ends Gauss–Newton | 1e-5 |
| `gl_data_scaling` | as above, plus `"auto"` | `"auto"` |

Validation (tests/test_group_lasso.py): stacked datasets equal one hand-stacked
operator; a linear method given as a simulation takes the Gauss–Newton path
and reaches the linear solution in ≤ 3 steps; gravity + DC recovers the
conductor (log₁₀σ −1.3 in the body for −1, −2.00 outside) and never increases
the objective; the cross-gradient matches SimPEG, lowers C monotonically in
λ3 and is unit-free.

## Bounds

`GroupLassoProblem(..., bounds=[(lower, upper) per model])`: scalars, per-cell arrays or None,
in each model's physical units.  With m = ref + w ζ / c (w, c > 0) they are a box on ζ, and the
box goes into the s update, which stays exact: per cell it minimizes
μ/2‖s − q‖² + λ1‖s‖ + ½λ2‖s‖² over lo ≤ s ≤ hi.  With a = μ + λ2, q′ = μq/a and κ = λ1/a the
minimizer (unique, the problem is strictly convex) is 0, when the box holds 0 and ‖q″‖ ≤ κ (q″
without the components that point out of the box at 0, e.g. a negative susceptibility against a
lower bound of 0), or s(r) = clip(q′ r/(r + κ), lo, hi) at the one root of r = ‖s(r)‖, found by
bisection over the cells at once.  So ADMM keeps two blocks, the same μ, factors and stopping;
no bounds, the same code as before.  λ1,max counts only the components of Zᵀb that point into the
box; the KKT check allows the normal cone of the box; a box without the reference (lower bound
above it) has no empty cells.  Checked (tests/test_group_lasso.py, TestExactBounds) against a
numerical minimum per cell (four boxes, one without 0), and against FISTA with the same bounded
step on the test problem with χ ≥ 0, 0 ≤ ρ ≤ 0.2 (upper bound active): the same objective to
10 digits, models within 2e-6 of the largest value, KKT residual 4e-7.

Worker: a model's bounds are its first dataset's `joint_regularizations` bounds_lower /
bounds_upper, else the task's `bounds_lower` / `bounds_upper`; the result's `group_lasso.bounds`
lists them.  (Geology constraints, with per-cell bounds, apply to single inversions only.)

## Field data: cell weighting, the balance of the datasets, over-relaxation

The first field test (Karnataka, 70 × 70 km, one mesh with terrain and 20 km padding; joint
report of 30 September, Section 5) showed three problems the synthetics had not.  On the 2 km
mesh (1,296 + 1,296 data, 21,575 cells, errors 0.5 mGal and 5 % + 10 nT, bounds −0.2…0.5 g/cc
and 0…1 SI, λ2 = 0.3):

| run | outside the core (ρ / χ) | χ²/N (gravity / magnetics) |
|---|---|---|
| paper settings, errors as data scaling (γ = 2) | 67 % / 82 % | 0.38 / 2.62 |
| γ = 1 | 49 % / 83 % | 0.37 / 1.63 |
| volume × depth weight (β = 1) | 8 % / 23 % | 0.45 / 2.44 |
| … and λ2 = 0.03 | 8 % / 20 % | 0.35 / 1.66 (ADMM at its 3000-iteration limit) |
| … and the balance (λ2 = 0.3) | 12 % / 33 % | 0.96 / 1.02 |
| sensitivity × cell volume, balanced (`sensitivity_volume`) | 28 % / 46 % (mostly beside the core; 2 % / 6 % below it) | 0.96 / 1.07 |
| SimPEG sparse, no coupling (for comparison) | 9 % / 36 % | 0.93 / 0.81 |

**Where the model goes.**  With γ = 2 every column of X has unit norm, so every cell is equally
cheap however little the data see it, and a wide padding soaks up the model.  `cell_weights`
(worker: `gl_weighting="depth"`) weighs each cell in the penalty like a SimPEG regularization
with depth weighting weighs it, R = volume × (z + z0)^(−β) with the depth exponent of the
model's own regularization (`depth_weighting_exponent`, β = 1 in these runs): the group norm
becomes Σ R_k ‖(c_p m_pk)‖, the discretized ∫ w(z) ‖m‖ dV, and large, deep cells cost what
their volume says.  w = 1/R is scaled so that the columns of X have a mean square of 1, as with
γ = 2, so λ2 and μ keep their meaning.

**The synthetic tests** (examples/output/coupling_comparison: three dense bodies with χ = 0.05,
0.01 and 0 SI, 357 stations, a mesh with padding) are where the paper's settings did best of all
couplings; the depth weighting does not carry over to them unchanged:

| blocks (true ρ 0.3, χ 0.05 / 0.01 / 0) | body A ρ / χ | ρ rms error | χ rms error | χ² (N = 357) |
|---|---|---|---|---|
| paper settings (γ = 2, amplitude balance, L-curve) | 0.267 / 0.049 | 0.0342 | 0.00438 | 382 / 436 |
| paper weighting + errors + balance | 0.260 / 0.048 | 0.0304 | 0.00393 | 328 / 390 |
| sensitivity × volume + balance | 0.260 / 0.048 | 0.0302 | 0.00396 | 326 / 391 |
| depth weighting β = 1 + balance | 0.229 / 0.030 | 0.0390 | 0.00294 | 344 / 369 |
| depth weighting β = 2 + balance | 0.265 / 0.046 | 0.0449 | 0.00379 | 396 / 335 |
| SimPEG L1–L2, cross-gradient (for comparison) | 0.095 / 0.019 | 0.0380 | 0.00356 | 394 / 399 |

(dipping bodies: ρ rms 0.0451 paper, 0.0423 paper + balance, 0.0422 sensitivity × volume,
0.0480 depth β = 1.)  The balance helps everywhere.  The weighting is a property of the mesh:
the group norm counts anomalous *cells*, so one large padding cell that explains a long
wavelength costs as much as one core cell; on the synthetic mesh the model hardly reaches the
padding and the paper's weighting is best, on the Karnataka mesh (20 km of padding, cells to 51
times the core volume) it is not.  `sensitivity_volume` (`cell_factors` = volume / smallest
volume, on top of γ = 2, not rescaled) keeps the paper's weighting in the equal core cells —
identical results on the synthetics — and charges a padding cell for its volume: on Karnataka it
keeps the model out of the bottom padding (2 % / 6 % below the core) but not out of the lateral
padding next to the core (30 % / 43 % of |model| × volume by the joint report's measure), where
cells are barely larger.  The depth weighting
keeps the Karnataka models in the core like the SimPEG runs but, with β = 1, underestimates the
compact magnetic body of the synthetic (0.030 SI of 0.05).

**One λ1 for two datasets.**  Along the sweep the gravity data are fitted faster than the
magnetic data (at the smallest λ1: 0.45 and 2.44 N), and the L2 term caps the magnetic fit, so
χ² = N in total can never be reached, or hides an overfitted gravity and an underfitted
magnetic model.  `gl_balance` gives each model a data weight: `GroupLassoProblem.scale_models`
multiplies its data scales and c_p by f_p (the operators, their factorizations and μ stay),
and keeps the group norm and the L2 term those of the same physical models by per-model
weights in the shrink (group weights 1/w_p, L2 factors 1/w_p², w_p the accumulated factors;
the bounded shrink takes them exactly: s_p(r) = clip(μ q_p r / (a_p r + λ1 g_p²), lo, hi)),
so only the data terms change — a data weight per dataset, as SimPEG's.  Each round sets
f_p = sqrt(χ²_p / N_p) (normalized, at most 4× per round) and finds λ1 for the total χ² = N
again by a secant in log λ1, warm-started from the same physical models; it stops when every
χ²_p / N_p is within 1/1.2…1.2 or a round does not narrow the spread.  On the 2 km mesh: two
rounds, data weights 0.60 (gravity) and 1.68 (magnetics).  Checked against FISTA on the
weighted objective (`test_scale_models_weighs_the_data_only`).

**Cost at full resolution.**  The coupled run with the balance took 124 min on c5.18xlarge
(the control 66): about 0.7 s per ADMM iteration against 0.25 s, because the exact bounded
shrink bisected every cell in single-threaded numpy (with the balance's unequal weights there
is no closed form).  The shrink now runs per cell in parallel with numba (`NUMBA_SHRINK_MIN`;
numpy below it or without numba; the two agree to 1e-14), 6.5 against 28 ms for 2 × 335,518
values on 15 cores.  Two shortcuts carry a 2 km run over: `gl_data_weights` starts the balance
from its weights (0.595 / 1.682 at 2 km, 0.625 / 1.600 at full resolution), and
`gl_lambda1_selection="search"` replaces the sweep by a descent to λ1,max / 10 and a secant
search for χ² = N from `gl_lambda1_ratio` × λ1,max (the ratio does not carry over as well: 8.8e-4
at 2 km, 2.0e-3 at full resolution, which the secant absorbs).  On the 2 km mesh: sweep and
balance 5,223 ADMM iterations in 93 s; the sweep from the weights 6,869 in 115 s (the sweep is
the cost); the search from the weights 2,711 in 54 s, the same models (82 % / 62 % in the core,
74 % of the magnetic cells anomalous in density), χ²/N 0.85 / 1.08 within the balance tolerance.

**Cost.**  Over-relaxation (`relaxation`, α = 1.6; Boyd et al. 2011, 3.4.3: the s and u updates
use α ζ + (1 − α) s_old) took 4,438 instead of 6,458 ADMM iterations over the 13-point sweep
(146 instead of 198 s), with the same χ² to five digits.  A smaller λ2 lets the data be fitted
without the balance, but ADMM then needs many more iterations for the same μ (3,000 per λ1 at
λ2 = 0.03, not converged).

## Not in this version

Spatial smoothness, adaptive μ,
uncertainty estimates; the upload page has no controls for model labels (the pipeline
accepts them) and offers the group lasso for gravity + magnetics; its λ3 is a manual
setting of the Inversion step.

## Using it from Python

```python
from geoinv3d.methods.group_lasso import JointGroupLassoProblem, from_simulations, plot_lcurve

# sensitivities K (N_f x M, nT per SI) and G (N_g x M, mGal per g/cc) on the same cells
P = JointGroupLassoProblem(K, G, f_obs, g_obs, std_f=sf, std_g=sg)   # or from_simulations(msim, gsim, f_obs, g_obs)
lc = P.lcurve(lambda2=0.3)                      # λ1 sweep with warm starts
res = P.solve_at(lc, lc.lambda1_corner, 0.3)    # or P.solve(lambda1, lambda2)
res.beta_physical, res.rho_physical, res.converged, res.primal_residual_history
plot_lcurve(lc)

# any models and datasets, nonlinear ones by Gauss–Newton, with a cross-gradient
from geoinv3d.methods.group_lasso import GroupLassoData, GroupLassoProblem
P = GroupLassoProblem([GroupLassoData("gz", g, 0, sg, operator=G),
                       GroupLassoData("gzz", gzz, 0, szz, operator=Gzz),
                       GroupLassoData("dc", d, 1, sd, simulation=dc_sim)],   # log-conductivity
                      model_names=["density", "log_conductivity"],
                      references=[None, np.full(n, np.log(1e-2))], mesh=dmesh)
res = P.solve(0.05 * P.lambda1_max(), 0.3, cross_gradient=0.1)
res.model("log_conductivity"), res.prediction("dc"), res.gauss_newton
```
