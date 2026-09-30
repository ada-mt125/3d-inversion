# Joint inversion: the coupling as a choice of its own

Code: `geoinv3d/methods/coupling.py` (the registry, the terms, PGI's units),
`geoinv3d/methods/joint.py` (`JointInversion(coupling=…)`, PGI's build),
`geoinv3d/methods/directives.py` (`CouplingScale`), `geoinv3d/methods/group_lasso.py`
(the group lasso's own solver), tests: `tests/test_coupling.py`.

A joint inversion minimizes

    Σ_d φ_d(m)  +  β [ Σ_i φ_reg,i(m_i)  +  w · φ_coupling(m_1, …, m_P) ]

— the data misfits, a **regularization of each model** and a **coupling between the
models**. The last two are separate choices (Colombo & Rovetta 2018; Haber & Holtzman
Gazit 2013): the regularization says what each model should look like (smooth, compact,
focused), the coupling what the models have in common. Before this layer the group lasso
was listed as a "regularization type" next to l2 and sparse, and the cross-gradient was
a weight; now `coupling` is its own setting and its own column of the results tree.

## The couplings

| `coupling` | family | assumes | term | solver |
|---|---|---|---|---|
| `cross_gradient` | structural | boundaries coincide, values and signs free | Σ‖∇m_i × ∇m_j‖² (SimPEG `CrossGradient`, every pair) — Gallardo & Meju (2003, 2004) | Gauss–Newton |
| `joint_total_variation` | structural, convex | the models change in the same places, sparsely | Σ √(Σ_i s_i²‖∇m_i‖² + ε) (SimPEG `JointTotalVariation`, models scaled, below) — Haber & Holtzman Gazit (2013) | Gauss–Newton |
| `linear_correspondence` | petrophysical | a linear relation λ1 m1 + λ2 m2 + λ3 = 0 in every cell | Σ (λ1 m1 + λ2 m2 + λ3)² (SimPEG `LinearCorrespondence`; two models) | Gauss–Newton |
| `pgi` | petrophysical | every cell is one of a few rock units (a Gaussian mixture of the properties) | SimPEG `PGI`: mixture smallness + each model's smoothness — Astic & Oldenburg (2019), Astic et al. (2021) | SimPEG's PGI directives |
| `group_lasso` | sparsity | anomalies share their cells, any sign or ratio | λ1 Σ_k ‖(m_1k, …, m_Pk)‖₂ + ½λ2‖m‖² — Utsugi (2025) | ADMM (its own) |
| `none` | — | nothing: one β for all models | — | Gauss–Newton |

`group_lasso` with `gl_cross_gradient` (λ3) > 0 adds a cross-gradient to the group lasso:
a hybrid that is **not** Utsugi's method, recorded as `group_lasso+cross_gradient` and
shown as "group lasso + cross-gradient (hybrid)".

The Gauss–Newton couplings add their term(s) to the per-model regularizations of
`JointInversion` (any of l2, sparse, L1–L2 IRLS, MGS, TV per model). PGI replaces the
per-model regularizations with its own (the regularization settings are not used).

## A unit-free coupling weight

The coupling terms come in the units of the models' products. The cross-gradient is
quartic in the model amplitudes: for a density in g/cc and a susceptibility in SI it is
~10⁻¹⁰ at the inverted models, and on the block synthetic of
`tests/test_joint_regularization.py` (gz + TMI, 49 stations) raw weights of 1, 10³, 10⁶
and 10⁹ all gave the uncoupled answer (correlation of the two models 0.897–0.899). **The
page's former default, cross-gradient weight 1, therefore coupled nothing.**

`CouplingScale` (a directive) makes the weight unit-free: the coupling is off in the
first iteration (at the starting model the cross-gradient has no curvature); after the
first and the third, each coupling term's multiplier becomes

    w × λmax(regularization Hessian) / λmax(coupling-term Hessian)

at the current model, so **w = 1 weighs the coupling like the models' regularization**.
The group lasso's λ3 is normalized the same way against its data term. On the block
synthetic (χ² of both datasets kept near N = 49):

| cross-gradient w | 0 (none) | 0.1 | 1 | 10 |
|---|---|---|---|---|
| correlation of ρ and χ | 0.90 | 0.93 | 0.97 | 0.99 |

`coupling_options: {"scale": "raw"}` keeps the old behaviour (the weight as a plain
multiplier).

**Joint total variation** adds the models' squared gradients, so the model with the
larger gradients in its own units decides alone where the structure is: unscaled, w = 10
left gravity at χ² = 92 and fitted magnetics to χ² ≈ 0. Each model is therefore scaled by
s_i = 1 / rms(∇m_i) (set with the weight); both datasets then fit at w = 0.1, 1 and 10.
A total variation's curvature is 1/√ε where the models are flat, so with sparse (L1–L2)
models its largest curvature was huge and the curvature-matched weight ~0 (JTV did
nothing). JTV is therefore weighed by **value** (w × φ_reg / φ_JTV), once, after the first
iteration (later the IRLS weights shrink φ_reg and the ratio kept strengthening the
coupling until the data were no longer fitted), and its ε is set to 1 % of the scaled
models' mean squared gradient.

**Linear correspondence** takes the relation as `slope` and `intercept` (model 1 = slope ×
model 2 + intercept, in model units: density contrast, SI, log conductivity) or as
`coefficients` [λ1, λ2, λ3]. With ρ = 6χ on the block synthetic the models correlate 1.00.

## PGI: rock units

`coupling_options`:

```json
{"units": [{"name": "BIF", "means": {"gravity": 0.6, "magnetics": 0.05},
            "stds": {"gravity": 0.06, "magnetics": 0.01}, "proportion": 0.03},
           {"name": "mafic", "means": {"gravity": 0.3, "magnetics": 0.01}, "proportion": 0.1}],
 "learn": false}
```

Means and spreads per model (keyed by model label — `gravity`, `magnetics`,
`dc_resistivity`, `mt` — or listed in model order), in model units; spreads left out are a
tenth of the unit's contrast. A **background** unit at the models' references is added
unless one is named "background", with a **tight** spread: 2.5 % of the largest unit
contrast of each property (SimPEG's joint tutorial uses ~3 %). This matters: on a
16 × 16 × 8-cell synthetic with a 4 × 4 × 3 block (0.3 g/cc, 0.05 SI, 144 stations):

| background spread | block cells found (of 48) | mean ρ in the block (true 0.3) |
|---|---|---|
| L2, no PGI | — | 0.045 |
| 10 % of the contrast | 0 | 0.047 |
| 1.7 % | 47 | 0.175 |
| 2.5 % (default) | 46 | 0.246 |

A wide background lets the smeared model count as background everywhere, so no cell is
ever assigned to a unit. `learn: true` lets the units' means move with the models (SimPEG's
`kappa` = 0; the background stays fixed): with a tight background the means collapsed onto
the smeared model (0.035 g/cc) and 444 cells were called "block" — learning is off by
default. The result carries `pgi.membership` (each cell's unit) and the units' final
means; SimPEG reorders the mixture's components, so each is named by the given unit
nearest to its means (by position, units of equal share came back under each other's names). Directives, as SimPEG's joint PGI tutorials: `AlphasSmoothEstimate_ByEig`
(`alpha_smooth_ratio`, 1e-2), `ScalingMultipleDataMisfits_ByEig`, `BetaEstimate_ByEig`
(`beta0_ratio`, 1e-2), `PGI_UpdateParameters`, `MultiTargetMisfits` (`chi_small`, 1),
`JointScalingSchedule`, `PGI_BetaAlphaSchedule`, `PGI_AddMrefInSmooth`,
`UpdatePreconditioner`; sensitivity weights per model from its own data.

On the upload page the units are typed one per line —
`BIF; density 3.3 ±0.1; susceptibility 0.05 ±0.01; share 0.03` (densities as rock
densities, the Model step's reduction density subtracted; resistivities in Ω·m with a
factor spread, `resistivity 10 ×2`) — or filled from the Model step: every body and layer
with a value for each property of the job becomes a unit.

## Where it shows

- Pipeline `params_json`: `coupling`, `coupling_weight` (a manual setting; Auto uses 1),
  `coupling_options`. Former keys still work: `regularization_type: "group_lasso"` means
  the group lasso coupling, `cross_gradient_weight` the cross-gradient at that weight.
- `InversionTask`: `joint_coupling`, `coupling_weight`, `coupling_options` (and the
  `coupling` property that resolves the former keys); packed with the task.
- `JointRegularizedInversionNode(coupling=…, coupling_weight=…, coupling_options=…)`.
- Results: `coupling` (kind, label, family, reference, weight, the multipliers set by
  `CouplingScale`), `pgi` for PGI; `settings.coupling` is the first column of the
  workflow tree for joint runs ("coupling: cross-gradient", …), then the regularization.
- Upload page, Inversion step: "How the models are coupled" — the coupling, its weight
  (manual), the linear relation or the rock units, and a description with the reference;
  the method list has one "Joint — A + B" entry per combination. The group lasso is
  offered for gravity + magnetics; its λ3 (manual) makes the hybrid.
- Upload page, Inversion step (manual), "Regularization per model": for each model its
  own regularization (“As above”, L2, Lp with its norms, L1–L2 with α, MGS / TV with the
  focusing percentile), α_s and bounds, sent as the dataset's `regularization`
  (`joint_regularizations`); empty fields keep the settings above. The group lasso and
  PGI have their own regularization, so there it offers the bounds only. The property
  bounds above apply to every model, the group lasso's included.

## References

- Astic, T. & Oldenburg, D. W. (2019). A framework for petrophysically and geologically
  guided geophysical inversion using a dynamic Gaussian mixture model prior. GJI 219.
- Astic, T., Heagy, L. J. & Oldenburg, D. W. (2021). Petrophysically and geologically
  guided multi-physics inversion using a dynamic Gaussian mixture model. GJI 224.
- Colombo, D. & Rovetta, D. (2018). Coupling strategies in multiparameter geophysical
  joint inversion. GJI 215(2), 1171.
- Gallardo, L. A. & Meju, M. A. (2003, 2004). Characterization of heterogeneous near-surface
  materials by joint 2D inversion of DC resistivity and seismic data (GRL 30); Joint
  two-dimensional DC resistivity and seismic travel time inversion with cross-gradients
  constraints (JGR 109).
- Haber, E. & Holtzman Gazit, M. (2013). Model fusion and joint inversion. Surveys in
  Geophysics 34, 675–695.
- Utsugi, M. (2025). Joint inversion of magnetic and gravity data using group lasso
  regularization. Earth, Planets and Space 77, 146.
- Vatankhah, S. et al. (2020). Generalized Lp-norm joint inversion of gravity and magnetic
  data using cross-gradient constraint (per-model sparsity + a cross-gradient coupling).
- Zhdanov, M. S., Gribenko, A. & Wilson, G. (2012). Generalized joint inversion of
  multimodal geophysical data using Gramian constraints. GRL 39.

## A demonstration: three dense bodies, high / low / no susceptibility

`examples/output/coupling_comparison/` (`run_coupling_comparison.py blocks|dipping`,
`make_report_figures.py`, `make_report_html.py`): six couplings, L1–L2 per model, on blocks
and on dipping intrusions; results, comparison tables, workflow viewers and a report.
Blocks, body means Δρ (true 0.3) / χ (true 0.05, 0.01, 0): no coupling 0.05–0.06 and χ of B
lost; group lasso 0.25–0.27 with χ 0.049 / 0.009 / 0.000; linear correspondence (Δρ = 6χ)
pulls C's density to 0.03 and gives it χ; PGI (true units) 0.13–0.22 but overfits.
