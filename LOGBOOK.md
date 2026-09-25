# Development Logbook

Chronological record of design decisions and implementation progress.

---

## 2026-09-23 — Project Inception

### Motivation

Building a 3D geophysical joint inversion framework in Python that:
1. Tracks every parameter change via a DAG (Directed Acyclic Graph)
2. Uses SimPEG for forward modeling and inversion (not reinventing the wheel)
3. Makes inversion workflows fully reproducible and inspectable

### Design Decisions

**DAG Architecture (from jif3d_visualization)**

Studied the DAG system in `jif3d_visualization` (TU Berlin, Max Moorkamp):
- `graph/node.py` — Node base class with `_compute`, `params`, `from_params`, `evaluate`, `invalidate`
- `graph/serialize.py` — JSON serialization with Kahn's topological sort for deserialization
- Key insight: "editing appends" — every parameter change adds a new node or invalidates the chain, never mutates

Adapted the core pattern but simplified for a pure-inversion focus:
- Removed Qt/UI thread requirements (`require_main_thread`)
- Removed `PayloadBudget` (memory management for large visualizations)
- Removed `EvaluationPlan` (background thread evaluation)
- Kept: `Node`, `Graph`, `register_node`, `invalidate` propagation, serialization

**SimPEG as Physics Backend**

SimPEG provides all the heavy lifting:
- `discretize.TensorMesh` for mesh generation
- `potential_fields.gravity/magnetics` for potential field methods
- `electromagnetics.static.resistivity` for DC
- `regularization.WeightedLeastSquares` for smoothness regularization
- `optimization.InexactGaussNewton` / L-BFGS for optimization
- `inverse_problem.BaseInvProblem` for combining data misfit + regularization

Our framework wraps SimPEG's API into clean `MethodBase` subclasses and
DAG nodes, adding parameter tracking and workflow persistence.

**Immutable Data Model**

Following jif3d_visualization's invariant: "input data is immutable."
All data payloads are frozen dataclasses with read-only NumPy arrays.
Transformations return new objects via `dataclasses.replace()`.

### What Was Built

```
geoinv3d/
├── core/         ✅ DAG engine (Node, Graph, serialize)
├── datamodel/    ✅ Mesh3D, PhysicalModel, SurveyData, InversionResult
├── methods/      ✅ Gravity, Magnetics, DC Resistivity, Joint
├── nodes/        ✅ Input, Transform, Forward, Regularization, Inversion, Export
├── io/           ✅ NumPy/JSON model persistence
├── viz/          ✅ DAG plots, model slices, convergence, 3D
├── examples/     ✅ Gravity forward + inversion example
└── tests/        ✅ Core DAG + serialization tests
```

### Known Limitations (to address in future sessions)

1. **DC Resistivity survey setup** — electrode array construction from SurveyData is simplified; needs proper dipole-dipole / Wenner array support
2. **Joint inversion node** — currently returns configuration dict; actual joint optimization loop not yet wired
3. **Cross-gradient coupling** — node exists but not yet connected to SimPEG's cross-gradient regularization
4. **No NetCDF I/O** — only NumPy/JSON for now; jif3D-compatible NetCDF format planned
5. **No MT/Tomography** — only potential fields and DC for now

---

## 2026-09-23 — SimPEG Integration Complete

### Import Migration

SimPEG 0.25.2 renamed its package from `SimPEG` to `simpeg` (lowercase).
Mixing old and new imports caused a `TypeError: data must be an instance of
Data, not Data` because the two namespaces create separate class identities.

Fixed all 6 method/node files to use `from simpeg import ...`.

Also fixed:
- `IterationCollector` now inherits from `simpeg.directives.InversionDirective`
  (SimPEG validates directive types at runtime)
- `maxIterCG` deprecated → `cg_maxiter`

### Synthetic Gravity Inversion — Verified

`examples/synthetic_gravity_inversion.py` runs end-to-end:
- 15x15x8 mesh (1800 cells), 64 surface stations
- Buried block anomaly at +0.5 g/cm³
- 7 Gauss-Newton iterations, converged (phi_d = 25.74)
- Recovered model range: [-0.012, 0.182] vs true [0.0, 0.5]
- Full DAG serialized with 8 nodes and Mermaid diagram

---

## 2026-09-23 — Interactive DAG Viewer

### Interactive Web Page

Built `geoinv3d/viz/dag_interactive.html` — a standalone HTML/JS viewer that:
- Loads `.geoinv3d.json` workflow files via drag-and-drop or file picker
- Renders the DAG as an interactive SVG with automatic layered layout
- Color-codes nodes by type (green=input, blue=model, orange=survey, purple=forward,
  teal=regularization, red=inversion, light green=export)
- Inversion nodes rendered as diamonds, input/survey as rounded rectangles
- Click any node to see full parameters in the sidebar panel
- Shows connections (inputs/outputs) with clickable navigation
- Summarizes large arrays (model values, locations) as statistics
- Supports pan (drag), zoom (scroll), fit-view, and SVG export

### Python Helper

`geoinv3d/viz/serve_dag.py` generates a self-contained HTML file with
workflow data embedded — no server needed:

```python
python -m geoinv3d.viz.serve_dag workflow.geoinv3d.json --serve
```

Or programmatically:
```python
from geoinv3d.viz import generate_viewer
generate_viewer("workflow.geoinv3d.json")
```

---

## 2026-09-23 — Process Data Storage & Convergence Viewer

### Enhanced Serialization

`core/serialize.py` now supports `include_outputs=True`:
- `graph_to_dict(graph, include_outputs=True)` exports each node's computed output
- For inversion nodes: full iteration history (phi_d, phi_m, phi_total, beta, model
  statistics per iteration), convergence status, final model summary
- For other nodes: type-specific summaries (mesh shape, model stats, survey info)
- `save_workflow(graph, path, include_outputs=True)` convenience function
- Workflow JSON gets a `created` timestamp

### Interactive Viewer (v2) — Inspired by jif3d_visualization

Rebuilt `dag_interactive.html` with three tabs:

**DAG Graph tab** (enhanced):
- Green dot indicator on nodes that have computed output data
- Sidebar shows full output summary when clicking a node (convergence info,
  final model statistics, method details)
- Timestamp display in stats bar

**Convergence tab** (new, inspired by jif3d_visualization's convergence window):
- Canvas-rendered phi_d + phi_m chart (like jif3d's RMS per data objective panel)
- Total objective + beta chart with dual axes
- Model statistics chart (min/max/mean per iteration with filled range)
- Deferred rendering: charts only draw when tab is visible (fixes canvas sizing)

**Iterations tab** (new):
- Slider to browse any iteration snapshot
- Metric cards: phi_d, phi_m, phi_total, beta
- Model statistics: min, max, mean, std
- Change-from-previous: percentage deltas with colored arrows

### File-Interactive Design

The viewer is fully file-interactive:
- Loads `.geoinv3d.json` files via drag-and-drop or file picker
- `generate_viewer()` embeds workflow data (including process data) into a
  self-contained HTML file that works offline
- Responsive layout: sidebar hides on narrow viewports

### Next Steps

- Add cross-gradient regularization from SimPEG
- Add MT method wrapper (`simpeg.electromagnetics.natural_source`)
- Add NetCDF I/O for jif3D compatibility

---

## 2026-09-25 — Raw-Data Pipeline Honours the Upload Spec

`cloud/worker.py::run_data_pipeline` now consumes the upload page's `params_json`:

- **Datasets**: each `datasets[]` entry is loaded from all of its files (grids
  `.tif/.grd/.asc`, station data `.csv/.xyz/.txt/.dat/.obs/.npy/.npz`) and gets its
  own noise model `noise_pct·|d| + noise_floor`. Legacy `data_file` params still work.
  New readers: `read_station_table` (header / Geosoft-comment column detection),
  `read_ubc_obs`, `read_esri_ascii`, `read_npz_points`.
- **Components**: `GravityMethod(component="gz"|"gzz")`, `MagneticsMethod(component="tmi"|"bz")`.
  Method aliases (`magnetic`, `dc`) are mapped to canonical names.
- **Topography**: a DEM (grid or x,y,z points) drapes stations that have no
  elevation and deactivates cells above ground; otherwise `flat_elevation` sets the
  station height and the mesh top. Stations with their own z keep it.
- **Mesh** spans the union of all survey extents; `mesh_type="octree"` builds a
  `TreeMesh` refined along the ground surface (wrapped by `datamodel.DiscretizeMesh`).
- **Joint** jobs run `JointInversion` with `joint_weights`, `cross_gradient_weight`
  (auto: 1), active cells, and `alpha_s` + `alpha_x/y/z` as length scales (same
  convention as the sparse single-method path). Joint stays L2: norms/bounds are
  reported in `result["notes"]`.
- **Auto mode** ignores manual-only keys, so defaults are exactly the previous ones.
- Fixes: `make_simulation_active` used the removed `valInactive` argument (SimPEG ≥ 0.24)
  and the wrong map size; 3+-method cross-gradient terms now act on the full joint model.
- Results add `mesh` (discretize `to_dict`), `active_cells.npy`, `topography`, `datasets`.

Tests: `tests/test_data_pipeline.py` (plumbing, readers, end-to-end SimPEG runs on
~1000-cell meshes; no AWS).

---

## 2026-09-25 — L1–L2 (Elastic Net) Regularization

Implemented Utsugi (2019, EPS 71:73) as restated by Nwosu & Becken (2025, GJI 243 ggaf390):
`φ_m = (1−a)/2·‖m̃‖² + a·‖m̃‖₁`, `m̃_j = ‖g_j‖^½ (m_j − m_ref,j)`.

- `methods/regularization.py`: `ElasticNetSmallness` (a `SparseSmallness`) and `ElasticNet`
  (a `Sparse` with that single term, so SimPEG's `UpdateIRLS` drives it). The L1 term is
  minimised by majorize–minimize: IRLS weights `q = a/(s·√(f²+ε²)) + (1−a)` make
  SimPEG's `‖W f‖²` touch the elastic net at the current model with the same gradient.
- `methods/directives.py`: `ElasticNetSensitivityWeights` sets `‖g_j‖` (unnormalized,
  data units) as the paper's depth weighting. SimPEG's own weights are σ-weighted and
  max-normalized, which shrinks m̃ so far that the L1 term swamps L2 and `a` stops mattering.
- `DampedUpdateIRLS`: SimPEG's IRLS β controller can lock into a two-cycle when φ_d is
  steep in β (seen: φ_d alternating 149/320, target 169). Each direction reversal now halves
  the log-β step. Used only on the L1–L2 path.
- Worker: `regularization_type="l1l2"` + `l1_ratio` (manual mode only; auto unchanged).
  Joint jobs stay L2 and say so in `notes`. Upload page: Manual → Regularization select.
- Fix (all bounded sparse jobs): a start model on a bound (m0 = 0, lower = 0) put every
  cell in ProjectedGNCG's active set and the model never moved; m0 now starts 1e-4·range inside.

Validation: IRLS solution = exact coordinate-descent elastic net (glmnet-style soft
threshold) to 1e-6 with identical zero patterns for a = 1, 0.5, 0. On synthetic TMI,
raising `a` concentrates the anomaly (cells > 10 % of max: 587 → 131 → 60 for a = 0, 0.5, 1)
and raises the peak towards the true value, matching the papers' conclusions.

Next: reproduce the paper's synthetic tests, L-curve / GCV for λ, focusing (MGS) and TV,
then mesh extensions.

---

## 2026-09-25 — L1–L2 Paper Synthetic (Nwosu & Becken 2025 set-up)

`examples/l1l2_paper_synthetic.py` runs the paper's magnetic survey (TMI, B0 = 42000 nT,
I = 45°, D = 30°, 26 × 26 stations at 2 km height) over a χ = 0.03 SI prism
(10 × 10 km, 2–10 km deep — our geometry; the paper was not reachable from this
environment) on a 28 × 28 × 20 mesh of 2 × 2 × 1 km cells, through
`run_single_inversion`. 18 runs (L1–L2 a = 0.1/0.3/0.5/0.7, sparse, L2 × σ = 2/5/10 nT)
take ~2 min. Report: `examples/l1l2_paper_synthetic_report.md`.

- a controls compactness as in the paper: V(≥10 % of peak) 2208 → 740 km³ (true 800),
  peak χ 0.025 → 0.054 for a = 0.1 → 0.7 at σ = 2 nT. a = 0.3–0.5 gives the half-peak
  bottom at 9–11 km (true 10 km); a = 0.7 overshoots the peak and lifts the bottom.
- The worker's default sparse (norms 0,2,2,1, alpha_s = 1e-4) smears the body to 14–15 km;
  smooth L2 (no depth weighting) puts it at 0–3 km and over-fits (χ²/N ≈ 0.6).
- L1–L2 needs ~40–45 IRLS iterations here; the example uses max_iter = 60.
- Fix: `pyproject.toml` named a non-existent build backend and discovered `deploy/`
  as a package, so `pip install -e .` failed; now `setuptools.build_meta` + explicit
  package discovery.

---

## 2026-09-25 — λ Selection, Utsugi (2019) CDA, MGS/TV Focusing

**λ selection** (`methods/regparam.py`): `FixedBetaIRLS`, L-curve corner (cubic spline
curvature in log β), GCV with the exact influence-matrix trace (data-space form, cells at
bounds excluded). Worker: `beta_selection = auto | discrepancy | lcurve | gcv`. A sweep
must reuse one IRLS eps (and one MGS/TV e): with per-run eps the trade-off curve was not
monotone and the L-curve corner landed at χ²/N = 127.

**L1–L2 as in Utsugi (2019)** (`methods/l1l2_cda.py`, now the default `l1l2_solver="cda"`;
SimPEG IRLS remains as `"irls"`): coordinate descent with soft thresholding (Eq. 26),
optional bounds, warm starts along λ from λ_max (0.1 log10 steps, `lambda_decades`),
wS1/wS2 weighting (`l1l2_weighting`, default S2 as recommended by the paper), λ from the
L-curve of ‖f − Xβ‖ vs P(β; α); also discrepancy and GCV (elastic-net dof). Magnetics is
solved in A/m as in the paper. Active-set polishing (kept only if signs/bounds hold, a full
sweep still decides convergence) cuts α = 0.99 paths from >10 min to <1 min. numba is a
dependency now. Pitfall: the λ_max point has P ≈ 1e-14 and made the spline fake a corner;
points with P < 1e-6·max(P) are dropped.

**MGS / TV** (`methods/regularization.py::Focusing`, `regularization_type = mgs | tv`):
isotropic cell gradient, exact IRLS majorizer via face weights; e fixed at IRLS start
(95th percentile of |∇m| of the L2 model; ×1 MGS, ×0.1 TV).

**Paper-protocol synthetic** (report `examples/l1l2_paper_synthetic_report.md`, supersedes
the IRLS-based comparison above): ε(α) is minimal at α ≈ 0.8 for both weightings (paper:
0.96 wS1, 0.90 wS2). wS1 α = 0.8 recovers the prism best of all methods (peak 0.032–0.036
vs 0.03, half-peak depth 2–10/11 km, χ²/N ≈ 1, no bounds). Unlike the paper, wS2 is worse
here (anomaly pushed to 5–14 km, peak ×2.5), probably because this survey is 40× larger in
scale. MGS flips between collapse and no focusing as e changes by ×1.5; with χ ≤ 0.04 it is
stable and blocky. TV ≈ sparse (deep tail), lowest ε after L1–L2.

Tests: 139 (test_regparam, test_l1l2_cda, test_focusing added).

Next: mesh extensions (scope to be confirmed with the user).

---

## 2026-09-25 — Viewer Integration and λ-Selection Benchmark

**Viewer / upload page**: Manual mode offers L1–L2 (solver CDA/IRLS, weighting wS1/wS2,
λ range), MGS and TV (focusing parameter), and the trade-off choice (auto / χ² = N /
L-curve / GCV). `RegularizedInversionNode` runs any worker regularization as a DAG node;
its output carries `regularization` and `selection` (λ path or β sweep with each
criterion's pick and warnings). The Convergence tab draws the L-curve, χ²/N and GCV with
the picks marked. Checked in Chromium (charts, form toggles, submitted params via a stub
fetch, data fit, depth slices); the 3D tab needs Plotly from cdnjs (blocked in the sandbox).
Demo: `examples/regularization_viewer_demo.py` → `regularization_comparison.geoinv3d_viewer.html`.
Viewer 3D arrays run from the top down; discretize order is flipped on export.

**L-curve without a corner**: on small, easily over-fitted problems the curvature is ≤ 0
everywhere and the maximum only marks the flat end (χ²/N = 0.005). `lcurve_corner_info`
reports validity; CDA `auto` then falls back to χ² = N, otherwise a warning is shown.

**Benchmark** (`examples/lambda_selection_benchmark.py`, 3 models × 4 noise levels ×
L1–L2/sparse/L2, scored against the best swept model): with the true σ all criteria are
within 1.3× (discrepancy ≤ 1.09, GCV ≤ 1.22, L-curve ≤ 1.30); misjudging σ by ×2 costs up
to 1.72× and χ² = N is then unreachable in 5/12 cases. GCV picks χ²/N ≈ 0.64 but the best
models themselves sit at 0.74–0.84, so it is not worse in model error (earlier single-case
conclusion revised in the report).

Not run after the last changes: the full test suite (the user will test locally).
Last full run: 137 passed; since then the affected suites passed (regparam, l1l2_cda,
regularized_node, pipeline choices/positivity).

---

## 2026-09-26 — Data-Driven Mesh Design and a Step-by-Step Upload Page

**Mesh from the data** (`cloud/meshing.py`, mirrored as `GeoMesh` in `viz/dag_interactive.html`;
both implement the same rules and must be changed together; `tests/test_meshing.py`):
- Data spacing: grid spacing for grids; for stations the area per station
  (N s² = A + P s/2 + s² on the convex hull: exact for grids, L/(N−1) on a line), but at
  least half the across-line spacing (area² / nearest-neighbour median), so 5 m along-line
  airborne sampling does not ask for 5 m cells.
- Recommended: horizontal cell = nice(spacing), vertical = half, core depth and padding =
  half the survey width; cells are stepped up while > 500 k cells or the float32
  sensitivity matrix > half the instance RAM (8 GB on the server).
- The pipeline fills missing `core_cell_m` / `core_cell_z_m` / `depth_core_m` /
  `pad_distance_m` from the recommendation and reports `mesh_design`
  (source auto / user / mixed, per-dataset spacing, recommended vs used).
- Fix: padding cell count ignored the 1.3 expansion (100 m cells, 2 km padding → 20 cells
  reaching ~80 km); it now stops once the expanding cells reach the padding distance.

**Upload page as a wizard**: 1 Data (cards, topography, what to invert) → 2 Mesh (data
resolution read in the browser from .csv/.xyz/.txt/.dat, UBC .obs, Surfer 6/7/ASCII .grd,
GeoTIFF and .npz; recommended mesh, editable, live cell and memory estimate; instance type)
→ 3 Inversion (regularization) → 4 Review & submit. Changing the data re-locks later steps.
Browser parsers and recommendations were checked against Python on nine files in every
format and five recommendation cases (identical).

---

## 2026-09-26 — Job Monitoring (Upload Step 5) and AWS Fixes

**What Submit does**: the page POSTs each job to the local API (`python -m geoinv3d.api`,
127.0.0.1:8000). The API uploads the files and `params.json` to
`s3://<bucket>/geoinv3d/jobs/<task_id>/`, submits an AWS Batch job (image from the job
definition) and records the job in `~/.geoinv3d/jobs.json`. The worker downloads the data, runs
`run_data_pipeline`, and writes `progress.json` (stage, iteration, φd vs target; at most
every 10 s), then `result.zip` and `result.json`.

**Step 5 "Jobs"**: lists jobs with Batch lifecycle (Submitted → Queued/RUNNABLE → Starting →
Running → Done), stage and iteration progress, run time and cost estimate, worker log
(CloudWatch), Stop (confirmation), and result download; polls every 15 s while a job is active.

**Fixes**:
- Stop used `CancelJob`, which does nothing to STARTING/RUNNING jobs (the instance kept
  running and billing); now `TerminateJob`, and the job shows as Stopped.
- The worker exited 0 after an error, so failed inversions showed as SUCCEEDED; it now uploads
  the error and exits 1.
- The instance type chosen on the page was ignored; it now sets the job's vCPU/memory request
  (`INSTANCE_RESOURCES`).
- Job records lived only in server memory; the API listened on 0.0.0.0 (anyone on the network
  could start jobs with this machine's AWS credentials; now 127.0.0.1); upload file names are
  reduced to their base name (no `../`).
- Docker image lacked rasterio (GeoTIFF uploads).

**Deploying**: the worker code is baked into the Docker image, so every change under
`geoinv3d/` needs the image rebuilt and pushed to ECR before AWS runs it. Tests use fake
S3/Batch/Logs clients only (`tests/test_aws_jobs.py` refuses real boto3 clients).
