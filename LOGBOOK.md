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

---

## 2026-09-27 — EC2 Backend (the account allows EC2/EBS only)

The AWS admin scoped the user's keys to EC2/EBS/SSM and requires every created resource to carry
`Owner=<IAM user>` (miaozhou). DryRun probes showed the keys still had wider rights, but we stay
within the stated scope, so the Batch design (S3/ECR/Batch/IAM, `deploy/setup_aws.py`) is parked
until the admin approves it. Region: ap-south-1 (the account default); on-demand instances only.

`geoinv3d/cloud/ec2.py` (`EC2Backend`, default backend of the API server):
- one instance per job, Amazon Linux 2023, tagged Owner/Project/Name/geoinv3d:task on the
  instance, volume and network interface; key pair `geoinv3d-<user>` (private key in
  `~/.geoinv3d/keys/`) and security group `geoinv3d-ssh-<user>` (SSH from this machine's IP only);
- user data installs Python 3.13 and the pinned packages with uv, then marks READY;
- the API server uploads the code (tar of the package), data and params over SFTP and starts
  `python -m geoinv3d.cloud.worker --local` (writes progress.json / result.* on the instance);
- refresh reads progress over SSH; on `exit_code` it downloads the result to
  `~/.geoinv3d/results/<task>/` and terminates the instance;
- guards: shutdown-behaviour stop; stop if no job within 60 min, after 12 h of running, and 30 min
  after the job ends if nobody fetched the result (Jobs page: "Fetch result" restarts it).

Real run (`deploy/ec2_smoke_test.py`, c5.xlarge): launch → install (~30 s with uv) → upload →
4 smooth-L2 iterations → download → terminate in 93 s; χ²/N = 1.08; instance and volume gone.

---

## 2026-09-27 — Results in the DAG viewer, regional removal, gz sign convention

**Results → viewer.** `geoinv3d/viz/result_workflow.py` turns downloaded results (directory or
zip) into a DAG workflow: Mesh, Survey, optional True model, one node per run with iterations,
3D model (tensor meshes cropped to the core, octree sampled on a grid) and data fit.  The worker
now stores observed/predicted/std/regional data, the settings and the mesh design in the result.
Jobs page: "View in DAG" (`GET /api/inversion/{id}/workflow`) and "Download" next to Stop.
`deploy/ec2_multi_run.py` runs several regularizations of one dataset on parallel instances and
writes one combined viewer page.

**Regional field.** `geoinv3d/methods/regional.py`: least-squares polynomial trend surface (order
1–3, or the mean), removed per dataset (`"regional": {"method": "polynomial", "order": 2}`) before
the noise model; the removed trend and a summary go into the result.  Upload page: "Regional
field" select (gravity default order 1, magnetics none).

**Iso threshold.** The 3D view's iso surfaces are now scaled to each model's own range, not
the true model's peak, so a smooth recovered model (peak ≪ true) no longer vanishes when the
threshold is raised.

**gz sign convention (bug).** SimPEG's gz is the z-up component (negative over a dense body);
Bouguer/field data are positive down.  The pipeline passed field data to SimPEG unchanged, so real
gravity data would invert to a density model of the wrong sign.  Found in the Karnataka
pre-flight: the residual came out as −16.4..31.7 mGal, the mirror of the earlier workflow's
[−31.67, 16.44].  Fix: gravity gz is read as positive down by default and negated before SimPEG;
observed/predicted data are reported back in the file's convention; `gz_convention: "simpeg"`
(per dataset or global; upload page "gz sign") keeps data made with SimPEG's forward model as is.
gzz and TMI are unaffected.  The synthetic test data and `deploy/ec2_smoke_test.py` now make
positive-down gz.  219 tests pass.

---

## 2026-09-27 — Karnataka gravity on EC2; depth weighting; EC2/CDA fixes

**Runs** (AOI 641–711 km E, 1634–1704 km N; 5040 data at 1 km; order-2 trend removed;
84×84×27 = 190,512 cells): sparse on c5.2xlarge (4.6 min, χ²/N 0.97) and L1–L2 by IRLS on
c5.4xlarge (3.5 min, χ²/N 1.08).  Instances terminated; about $0.5 in total.

**Why the sparse model reaches the bottom of the mesh.** With alpha_x/y/z passed as length
scales, SimPEG weights the gradient terms by (1 × 500 m)², so alpha_s = 1e-4 makes the
smallness (p_s = 0) term ~2500× weaker: the model is smoothness-driven, and with p_z = 1 a
vertically constant column costs nothing.  50% of |mass| lay below the 10 km core (D50 9.1 km)
while the power spectrum gives source depths of 1.1 / 2.2 / 6.0 km.  A 15-run local study on a
2 km mesh (25 s each; the baseline reproduces the full-resolution result) showed: alpha_s = 1
cuts the share below the core to ~20%; Li & Oldenburg depth weighting with β = 1 also stops the
leak into the lateral padding (33% → 5%) and gives D50 4.6 km; β = 0.5 / 1.5 / 2 give D50
1.9 / 6.8 / 8.9 km.  Report: `examples/output/reports/karnataka_gravity_depth_report.html`;
all runs in one viewer: `examples/output/depth_study/karnataka_depth_study.geoinv3d_viewer.html`.

**New option** `depth_weighting: "sensitivity" | "depth"` with `depth_weighting_exponent` (β)
for l2/sparse/mgs/tv (manual mode; upload page "Depth weighting").  "depth" sets SimPEG cell
weights (z + z0)^(-β) (norm weight (z + z0)^(-β/2), z below the nearest station, z0 half the
smallest cell) instead of the UpdateSensitivityWeights directive.

**Fixes found on the real runs**
- EC2 start command: `a && b && nohup … & echo started` backgrounds the whole list, so the SSH
  channel stayed open until the worker ended; the read timed out after 60 s and the backend
  retried the set-up.  Now `{ nohup … & }`, plus a STARTED marker so a retried set-up follows the
  running job instead of starting it again.
- Code archive: directories from OneDrive come out read-only (r-x), so a second upload could not
  replace the code; entries now get modes 755/644, and the remote side `chmod -R u+w` first.
- L1–L2 CDA ran out of memory (31.6 GB on 32 GB): `invert_l1l2` copied G four times in float64.
  Now one Fortran-order copy scaled in place (peak ~11.5 GB here); `_polish` uses the Woodbury
  form when the free set exceeds the data count and skips when its copy would exceed 2 GB; CDA
  reports progress per λ point.  CDA is still single-threaded and each λ point took about twice
  as long as the previous one at this size (11/41 points in 11 min), so the run used IRLS.
- `deploy/ec2_multi_run.py --resume STAMP` follows existing instances; refresh errors are logged.
- `build_workflow` accepts runs on different meshes/data (own Mesh/Survey nodes).

---

## 2026-09-28 — alpha_s default 1; full-resolution check of the recommended settings

**Default.** `InversionTask.alpha_s` is now None = "the regularization's default": 1 where
alpha_x/y/z are length scales (l2, sparse, mgs, tv, joint), 1e-4 on the legacy smooth path with
raw SimPEG alphas (`effective_alpha_s`); the result's settings record the value used.  Upload page
default 1.  A 2 km check with the default sensitivity weighting: alpha_s = 0.1 still leaves 34%
of |mass| below the 10 km core (alpha_s = 1: 22%, 1e-4: 50%), hence 1.

**Full resolution on EC2** (1 km, 190,512 cells, c5.2xlarge, ~6 min and ~$0.03 each), p =
[0,2,2,2], depth weighting β = 1; profile under the main Bouguer high (671.5 E, 1664.5 N):

| run | χ²/N | core | below 10 km | D50 | half-max depth |
|---|---|---|---|---|---|
| original sparse (alpha_s 1e-4, sensitivity) | 0.97 | 24% | 50% | 9.1 km | 0.3–19.9 km |
| L1–L2 (IRLS) | 1.08 | 80% | 11% | 3.0 km | 2.3–4.8 km |
| alpha_s = 1, β = 1 | 0.96 | 83% | 15% | 4.6 km | 3.3–7.8 km |
| alpha_s = 0.1, β = 1 | 0.89 | 81% | 13% | 5.4 km | 3.8–9.3 km |

The columns are gone, but at full resolution the bodies sit deeper than on the 2 km study mesh
(alpha_s 0.1: 0.5–4.5 km there): the coarse study shows the trends, not the absolute depths.

**Fixes.** Parallel `start_job` calls raced to add the same security-group rule (the public IP had
changed) and one failed with InvalidPermission.Duplicate: key/SG set-up now runs under a lock and
tolerates a duplicate rule.  `_finish` terminated the instance before recording the outcome, so a
refresh in between reported "terminated outside GeoInv3D" although the result was downloaded;
the outcome is now recorded first.  231 tests pass (plus the new EC2 ordering test).

---

## 2026-09-28 — Upload page: area of interest, thinning, sweeps, comparisons; local-only API

**Area step (wizard step 2 of 6: Data → Area → Mesh → Inversion → Review → Jobs).** The data are
read in the browser (`GeoPreview`: GeoTIFF uncompressed/deflate, stripped or tiled, reading only
the sampled rows — the 240 MB Karnataka TMI grid in about 2 s; Surfer ASCII/6/7; station tables,
.obs and .npz with a value column) and drawn with Plotly.  An HTML window over the plot is dragged
to move it and by its corners/edges to resize it (a move slides back inside the survey, a resize
is cut at its edge); width/height/centre in km, "Make square", "Whole survey"; pan and scroll-zoom
keep it aligned.  Per data type a thinning choice (every k-th node / one station per cell, with
factors reaching round spacings such as 1 km for 37.5 m aeromagnetics, or a custom spacing).
Counts follow the worker exactly (Karnataka AOI: gravity 5,041 vs 5,040 read — one blank node —
and magnetics 4,900 vs 4,900).  The mesh step designs the mesh for the window and thinned data.

**Pipeline.** `aoi` is validated; `decimate_spacing_m` (per dataset or job-wide) thins grids by
whole strides per axis (`grid_strides`) and points to the first station per cell
(`thin_points`, cells from the window's south-west corner); the result records what was done.
`decimate_stride` still works for grids.

**Sweeps.** Manual mode can vary α_s, depth-weighting β, the norms p or the L1–L2 ratio: one job
per value (at most 8, within the 36-vCPU limit), submitted as one group.

**Jobs page.** Groups are shown together with "Compare in DAG" and "Save comparison"; finished
jobs can be ticked and compared freely.  Saved comparisons (~/.geoinv3d/comparisons) reopen in the
viewer or download as a stand-alone page.  API: `GET /api/workflow?ids=`, `POST/GET/DELETE
/api/comparisons`, `GET /api/comparisons/{id}/page`.

**API security.** The server can launch EC2 instances, so only pages on this machine may use it:
CORS allows local origins only, and because a multipart POST needs no CORS preflight, requests
with a foreign Origin header or a non-local Host (DNS rebinding) are refused with 403.  Extra
origins/hosts via GEOINV3D_ALLOWED_ORIGINS / GEOINV3D_ALLOWED_HOSTS.  Submissions that would
exceed the vCPU limit (GEOINV3D_VCPU_LIMIT or aws.json "vcpu_limit", default 36) get 409.

**End-to-end run from the page (2026-09-28, user approved).** API server started locally
(`py -m geoinv3d.api`, 127.0.0.1:8000; a foreign Origin got 403 on the real server).  In the page:
NGPM_BA.tiff dropped on the gravity card (0 % + 0.5 mGal, order-2 trend), 70 km window at
(676, 1669) km thinned to 1 km (5,041 data), tensor mesh 1 km × 500 m / 10 km / 20 km (190,512
cells, as in the scripted EC2 runs), sparse p = [0,2,2,2], α_s = 1, bounds [−0.2, 0.5], sweep of the
depth-weighting β = 0.5, 1.5 on 2 × c5.2xlarge.  Both jobs: submit → install → invert → collect →
terminate in about 6 min; 30 iterations, χ²/N 0.90 (β 0.5) and 0.86 (β 1.5).  "Compare in DAG" on
the group opened both in one workflow; "Save comparison" stored it and its page downloaded
(2.5 MB, data embedded).  Instances terminated, no volumes left; server stopped.

**Fixes from the run.** (1) While finishing, the job showed the stale stage "starting"; it now
shows "collecting" (downloading the result, then shutting down).  (2) The viewer's 2D depth slice
and sections were drawn upside down (north at the bottom; the surface at the bottom of the E–W and
N–S sections), because the plot guessed the axis direction from the edge order and the result's
z edges run from the surface down.  Larger values are now always at the top (north up, surface
up), whatever the order of the edges; the 3D view was right.  The result viewer pages in
examples/output were regenerated.

**Correction (2026-09-28).** The note above that "at full resolution the bodies sit deeper than on
the 2 km study mesh" compared different places (the strongest cell on the 2 km mesh, the main
Bouguer high on the 1 km mesh).  Under the same point (671.5 E, 1664.5 N), the 2 km study gave
0.5–4.5 / 3.5–7.5 / 4.5–10.6 km for β = 0.5 / 1 / 1.5 (α_s = 1) and the 1 km mesh 2.3–4.8 / 3.3–7.8 /
4.3–11.1 km: the base within 0.5 km, the top and the peak 1–3 km shallower on the coarse mesh.
The full-resolution β sweep also shows the non-uniqueness plainly: the peak moves 3.8 → 6.8 → 9.3 km
with χ²/N 0.86–0.96, and β = 0.5 matches L1–L2 (2.3–4.8 km).  Both reports now have a section on
this (Chinese and English artifacts republished; English PDF regenerated, 11 pages).

---

## 2026-09-28 — Synthetic magnetic test (Karnataka field), full resolution on EC2

`examples/synthetic_magnetic_karnataka.py`: TMI on a 1 km grid over the 70 × 70 km gravity area,
80 m above flat ground (the survey's flight height), IGRF 2020 at 15.09 N 76.64 E (ppigrf):
42,100 nT, I = 19.3°, D = −0.9° true = −1.4° grid; noise 2 % + 1 nT.  Model (SI): A shallow block
1–3 km, 0.03; B deep block 4–7 km, 0.04; C dyke 0.5–6 km dipping 45° east, 2 km thick, 0.05.
Forward-modelled on a 500 × 250 m mesh (magnetic cells only): TMI −172 … +210 nT.

Inversions (`ec2_multi_run.py synthetic-magnetic`, 4 × c5.2xlarge, 8–9 min, ≈ $0.19): mesh of the
gravity runs (190,512 cells), p = [0,2,2,2], α_s = 1, susceptibility in [0, 0.1].

| run | χ²/N | corr(true) | in bodies | A depth (2.0) | B depth (5.5) | dyke dip (45°) |
|---|---|---|---|---|---|---|
| sparse, sensitivity | 1.11 | 0.21 | 10% | 3.9 km | 8.5 km | 64° |
| sparse, depth β = 2 | 1.07 | 0.24 | 16% | 3.7 km | 8.0 km | 62° |
| sparse, depth β = 3 | 1.31 | 0.17 | 4% | 4.9 km | 8.6 km | 66° |
| L1–L2 (IRLS) | 1.00 | 0.60 | 24% | 3.9 km | 6.7 km | 65° |

(depths: centroid of recovered susceptibility × volume in each body's footprint.)  Every method
puts the bodies 1.5–3 km too deep, recovers about a third of the susceptibility, and steepens the
dyke; the shallow block becomes a hollow "bowl" in the sparse runs.  The classic magnetic depth
weighting β = 3 is the deepest.  A 2 km local study (1,296 data) was much worse (A at 5–7 km,
χ²/N 1.2–1.8): for magnetics a coarse mesh is not a stand-in for the full one, unlike gravity;
it does show the trend (smaller β → shallower; β = 1 put B at 5.5 km).  Viewer with the true model:
`examples/output/synthetic_magnetic/synthetic-magnetic-1790589414.geoinv3d_viewer.html`.


## 2026-09-28 — Report rewrite, folder tidy-up, launcher, 3D occlusion, branch tree in the DAG

**Report.** `examples/output/karnataka_gravity/scripts/build_reports.py` (+ `report_style.py`) rebuilds the
Chinese and English reports and the English PDF from `figures/numbers.json`; the comparison of the six
full-resolution runs comes first (fit, lateral structure, depth, what is robust), then the cross-check with
the 8 km deep SimPEG mesh and Tomofast-x, the 2 km study, and only briefly why the original α_s = 1e-4
settings give columns.  Depths are centroids (the α_s = 1 models sit at the density bound, so the "peak
depth" is a plateau): main high 3.4 / 3.8 / 5.6 / 6.4 / 7.6 km for β = 0.5 / L1–L2 / β = 1 / α_s = 0.1 /
β = 1.5; 5.3 km on the 8 km mesh.  The residual maps differ even though χ²/N does not (0.86–1.08): L1–L2
leaves a broad positive residual over the main high.  Republished to the same two artifact links.

**Folder.** `examples/output/depth_study/` (old 20-run viewer) went to the Recycle Bin; the new
`karnataka_gravity/scripts/build_workflow.py` rebuilds all 22 runs into `karnataka_gravity/depth_study/`.
The earlier report moved to `karnataka_gravity/archive/`.  The three example viewers were regenerated.

**Launcher.** `GeoInv3D.bat` (repo root) starts `py -m geoinv3d.api` on 127.0.0.1:8000 unless
`/api/health` already answers, waits for it, and opens http://localhost:8000/.  The server now serves the
upload page at `/` with `window.GEOINV3D_API = location.origin` injected, so the page talks to whichever
port it came from (test in `test_api_local.py`).

**3D view "clipping".** Two causes: (1) the positive and negative isosurfaces were 70 % translucent, and
WebGL blends translucent surfaces without sorting them by depth, so a blue body behind showed through a
red one in front; (2) the colour range was min–max (−0.2 … 0.5), whose midpoint is +0.15, so the outer
shells of positive bodies at the 20 % threshold (+0.1) were drawn light blue.  Models that change sign
now get a colour range centred on zero (`modelRange`, also in the 2D sections and colour bars).

Transparency with correct occlusion (second pass, same day): gl-plot3d draws translucent triangles after
the opaque ones with "over" blending (`ONE, ONE_MINUS_SRC_ALPHA`), no depth writes, in buffer order and
trace order — so the result is right exactly when triangles are drawn back to front.  The shells are now
Plotly's own isosurface meshes (3 per sign, from the threshold to 90 % of that sign's peak), merged into
one mesh3d trace; `m3dSortShells` orders their triangles by eye-space depth (view · model matrices of the
scene) whenever the view turns by > 2° (checked every frame, throttled to ≤ 25 % of the main thread), and
`m3dUpload` writes the re-ordered data straight into gl-mesh3d's triangle buffers (its layout: vertices
c, b, a; hover still maps through the original triangle ids).  Opacity follows the colour bar, faint near
zero and solid at ±max, times one overall opacity; the threshold (share of the colour bar's |max|) hides
values closer to zero; the colour bar shows the same fading.  Checks (headless Chrome, a red and a blue
body one behind the other): sorted order puts the front body on top from both sides, the old two-trace
rendering put blue on top from both; the fast upload matches Plotly's own render under real lighting
(max pixel difference 8/255).  Karnataka β = 1 at 1 km: 166k triangles, ~75 ms per re-sort.
The data-fit tab lost its duplicate residual map (the gridded one, as for observed/predicted, remains).

**Branch tree.** `build_workflow` arranges several runs as a tree with one setting per column: mesh and
data → regularization → α_s (L1 share for L1–L2) → depth weighting → norms (sparse only) → results.
Every run passes through every column that applies to it (a first version skipped constant levels and
collapsed single-run branches, so a column mixed different settings; dropped).  Other settings that
differ within a study get extra columns.  The viewer lays such workflows out left to right with column
titles, all results in the last column as cards (combination, χ²/N, RMS, model range); a branch node's
sidebar lists every run below it with its fit.  `serve_dag._compact_workflow` keeps the new `order` and
`branch` fields.  Tests in `test_result_workflow.py`.  The synthetic magnetic viewer was regenerated
with the intermediate version and is not yet on the column layout.

## 2026-09-29 — Workspaces in the upload page; worker warnings

**Workspaces.** The page served by the API server (GeoInv3D.bat) now shows a named workspace at the top
left: click the name to rename it, ▾ to switch, create or delete workspaces.  The server keeps them in
~/.geoinv3d/workspaces/<id>.json ({name, job_ids, ...}); a job submitted from a page showing a workspace
joins it (submit form field workspace_id), and GET /api/workspaces/<id>/workflow returns all its
finished runs as one workflow (the branch tree; runs on the same data share one data node), cached until
another run finishes.  The page polls it every 20 s, so finished runs appear by themselves; the address
carries ?ws=<id>, and without it the page opens the workspace opened last (or a new one).  The Jobs list
shows the workspace's jobs, or all jobs on this machine.  Tests in test_api_local.py (TestWorkspaces);
the page was checked in headless Chrome with the API answered by a mock (no server started).

**Worker warnings.** The first job submitted from the page logged (1) runpy's "geoinv3d.cloud.worker found
in sys.modules": geoinv3d/cloud/__init__.py imported the worker, which the instances also run with -m, so
it loaded twice; the package now imports execute_task / pack_result / run_data_pipeline lazily.  (2)
SimPEG's FutureWarnings for tolCG / maxIterCG: renamed to cg_maxiter and the CG tolerances, with a catch:
in SimPEG 0.25.2 the deprecated tolCG sets the ABSOLUTE tolerance for ProjectedGNCG (cg_atol, cg_rtol 0)
but the RELATIVE one for InexactGaussNewton (cg_rtol).  Following the warning's advice (cg_rtol) would
have changed the bounded runs; the new arguments reproduce the old settings exactly (checked).

## 2026-09-29 — A slow L1–L2 job; stop with the result kept; overfitting alarm; size check

**The job "stuck at iteration 29"** (i-0dfab7c4300916cc6, submitted from the page): L1–L2 with the
coordinate-descent solver on the unthinned 500 m grid, 19,517 data × 53,120 octree cells (1.04e9
entries).  Read-only checks on the instance: the worker alive at 416 % CPU, 14.3 GB of 15.6 GB in use,
no swap; the "iteration" is the λ-path point (30 of 41 done after ~25 min); point 30 took 5 min and
564 sweeps against 244 for point 29, and φd was already 7,190 against the target 19,517.  The remaining
path would likely have run past the 12 h instance limit.  Advice: IRLS for L1–L2, or thin to 1 km.

**Stop and keep the result.** The worker now watches out/STOP (IterationCollector.stop_check, set by
run_local_job): SimPEG inversions end after the current iteration (opt.stopNextIteration) with that
model; the L1–L2 coordinate descent checks before every sweep, drops the unfinished λ point and picks λ
from the points done (the criterion if it can, else χ² = N, else the last point); a β sweep keeps its
discrepancy-principle run.  Results carry converged = False and stopped_early {reason, at_iteration,
of}.  EC2Backend.request_finish writes the flag over SSH; POST /api/inversion/<id>/finish (409 unless
running).  The page's job card: "⏹ Stop & keep result" next to "■ Terminate" (which still discards).

**Overfitting alarm.** A running job whose φd falls below half its target (χ² = N) shows a red alarm on
its card, pointing at "Stop & keep result"; the progress line now says "λ point i of n" for the
coordinate descent.

**Size check** in the Review step: data × cells of the dense sensitivity (octree ≈ 0.9 of the core
cells, as measured), memory ≈ entries × 12 B + 1.5 GB for the coordinate descent (float32 G + float64
copy; 13.8 GB estimated vs 14.3 GB measured for the job above) or × 6 B + 0.5 GB for SimPEG's solvers,
against the instance's usable memory: red above it, amber above 75 %; for the coordinate descent above
3e8 entries a warning that it takes hours.  Buttons switch to SimPEG IRLS or the smallest instance that
fits.  Checked in headless Chrome with a mocked API (the job above: amber, 13.8 of 14.2 GB; with IRLS
6.6 GB).  Tests: TestStop (CDA), stop flag in run_local_job, request_finish, the API endpoint.


## 2026-09-29 — Joint gravity–magnetic inversion with L2 + group lasso (Utsugi 2025)

`geoinv3d/methods/group_lasso.py`: ½‖b − Zζ‖² + λ1 Σₖ √(βₖ² + ρₖ²) + ½λ2‖ζ‖² by ADMM (s = ζ, scaled
dual), sensitivity weighting γ = 2, data balance C = max|f|/max|g| (or 1/σ).  The ζ-update never forms
X, Y or a 2M × 2M matrix: Woodbury with one Cholesky factor of the N × N XXᵀ + μI per method, reused
for every iteration and every λ1 (CG when a factor would pass 4 GB or the operator is matrix-free);
float32 kernels are used as stored.  Group soft threshold vectorized, O(M).  Boyd primal/dual stopping.
L-curve over λ1 (log misfit vs log group penalty) with warm starts; `from_simulations` for SimPEG.
Worker: `regularization_type="group_lasso"` on a joint gravity + magnetic job (`run_group_lasso_joint`,
settings `gl_*`), per-dataset data in the result (`data_<method>.npz`), shown in the viewer as a density
and a susceptibility model.  Upload page: "Joint — Gravity + Magnetic (group lasso, Utsugi 2025)" with
its own settings, a λ2 sweep, and the memory estimate.  Details and validation: docs/group_lasso_joint.md.

Findings.  (1) Scaling the gravity rows of the operator by C as well as the data leaves ρ̃ in mGal-like
units against nT-like β̃; the group norm then weighs them ~700× apart and ADMM sat at zero.  A scalar
scale belongs to the model variable (physical ρ = w ρ̃ / C).  (2) μ near the mean eigenvalue of XᵀX
(M/min(N, M)) converged fastest; λ2 = 0 is slow (thousands of iterations; smaller μ helps).  (3) float32
products floor the ADMM residuals at ~1e-6 relative.  (4) λ2 on the 16 × 16 × 8 synthetic (true 0.3 g/cc):
0.01 → peak 1.4, 0.3 → 0.28, 1 → 0.16 and χ² stuck above N for small errors; default 0.3.
Tests: tests/test_group_lasso.py (43: prox, scaling, solvers, KKT + FISTA reference, A–E, L-curve,
pipeline end to end).  The paper's full text could not be fetched (publisher sign-in): the method follows
the specification we were given.

**Hand-off (session moved to another account).**  Committed on branch claude/cool-hawking-9zd59t.
Full test suite after these changes: 318 passed (8 warnings, all from SimPEG / discretize /
pymatsolver).  Next: restart the API
(GeoInv3D.bat) to use the new page option; nothing has been run on AWS with the group lasso yet
(the user approves every instance launch).

---

## 2026-09-29 — Joint inversion with any regularization; group lasso for many models, MT/DC and a cross-gradient

Environment: macOS (Apple Silicon), Python 3.13.15 in `.venv` (uv), SimPEG 0.25.2.  Before these
changes the suite gave 316 passed, 2 failed on this machine: `TestSolvers::test_solves_the_normal_equations[cg-cols0]`
and `test_per_row_scaling` miss their CG tolerance by < 2× (2.1e-6 vs 1.3e-6).  Not a solver bug:
with a float64 kernel CG is accurate to ~4e-9; with SimPEG's float32 kernel every CG iteration
multiplies by it (one product rounds at ~9e-8) and the rounding accumulates to ~1.6e-6 on Apple
Silicon (Accelerate), just under 1e-6 with MKL.  The Cholesky solves make their factors in float64
(8.7e-8).  Fixed in the tests: 1e-5 for CG with a float32 kernel (`FLOAT32_SOLVE_TOL`), 1e-6 kept
for Cholesky; the float32 kernel itself stays (memory).

**Method one (`methods/joint.py`).**  `ModelRegularization` per model: "l2", "sparse", "l1l2" (IRLS),
"mgs", "tv", or "smooth" (the old WeightedLeastSquares), each with its alphas, depth weighting
("sensitivity" / "depth" / "none") and bounds (ProjectedGNCG on the joint vector).  Without any, the
original path runs unchanged (tested).  `MethodSetup.model` labels: datasets with one label share a
model (gz + gzz of one density model; MT + DC of one conductivity model); unlabelled datasets keep a
model each ("gravity", "gravity_2", …), so results stay keyed by method name.  Worker: joint tasks
use `regularization_type` for every model, `joint_regularizations` (per-dataset overrides,
`JOINT_REG_KEYS`), `joint_models`, `joint_balance`; the upload pipeline takes `model` and
`regularization` per dataset (the latter in manual mode only) and applies task-level bounds to all
models.  Results carry per-model info, per-dataset χ² and `joint_data` (the viewer now shows method-one
joint runs too).

Findings.  (1) One beta for several models needs two things SimPEG does not do: `UpdateSensitivityWeights`
normalizes all models together, so the model with the smaller sensitivities is barely regularized
(`JointSensitivityWeights`: per model, same formula); and the models' penalties add up in their own
units (`JointRegularizationBalance`: multipliers from the ratio of the data and regularization
Hessians' largest eigenvalues per model, like BetaEstimate_ByEig).  (2) That start is not enough: L1–L2
ended at χ² = 103 and 1 for two datasets of N = 49 (total on target).  The directive now also moves
regularization from the better- to the worse-fitting model after every iteration (step in log capped
at 2×, geometric mean kept, only once χ² < 3N — before, χ²/N measures each dataset's signal).
All kinds then end at χ²_k ≈ N_k and, with a negligible coupling, correlate > 0.9 with their
single-method inversions.  (3) SimPEG's `UpdatePreconditioner` drops the combo multipliers;
`JointUpdatePreconditioner` applies them.

**DC and MT.**  `DCResistivityMethod` was a skeleton without a survey: now `SurveyData.locations` is
(n, 12) A/B/M/N rows (NaN B / N: pole source / receiver), consecutive rows with one source form one
SimPEG source (data keep their order), "volt" or "apparent_resistivity", log-conductivity model with
active cells (inactive: `sigma_inactive`), `storeJ`.  Half-space apparent resistivity 105–122 Ω·m for
100 on a 25 m mesh.  `MTMethod`: `make_simulation_mapped` passed the joint slice as conductivity while
`make_simulation` used ExpMap; now log-conductivity everywhere, with active cells; half-space ρa 97.8
for 100.  `MethodBase.default_model_value` (0, or log σ_background) and `linear`.

**Method two (`methods/group_lasso.py`).**  `GroupLassoProblem`: P models, datasets `GroupLassoData`
(linear: an operator; nonlinear: a SimPEG simulation), a model's datasets stacked (`StackedOperator`),
groups (ζ_1k, …, ζ_Pk), references (log σ_background: "empty" = background).  `JointGroupLassoProblem`
is its two-model case; on the paper's problem it gives bit-identical iterations, models and L-curve
to the previous code (checked against a copy of it, both scalings, Cholesky and CG).
- Cross-gradient λ3 C(ζ) between every pair: `CrossGradientTerm` (SimPEG's discretization; value and
  gradient equal to `CrossGradient` to 1e-12), on the anomalies without the depth weighting.  For fixed
  other models it is quadratic and PSD, so the ζ update solves one SPD system per model (Gauss–Seidel,
  PCG preconditioned by the Cholesky solve) and fixed points satisfy the KKT conditions of the whole
  nonconvex objective (tested).  Normalization: the pair factor makes the term's curvature at the
  damped least-squares models equal to the data term's (power iteration), so λ3 is unit-free (tested
  with density in g/cc vs kg/m³ and mGal vs µGal) and 1 weighs the coupling like the data; useful range
  ~0.01–1.  A first normalization by max|Xᵀb| was ~10⁶ too weak (λ3 ~ 1000 needed).  Inexact ADMM
  (CG tolerance following the primal residual) halved the time at default tolerances.
- Nonlinear datasets: Levenberg–Marquardt Gauss–Newton around ADMM (damping ν/2‖ζ − ζ_k‖² on the
  nonlinear models only, ν from the actual/predicted decrease ratio; the linear models keep their
  factors).  A plain backtracking line search stalled (steps of ¼ with the objective changing 1e-5 per
  step): near the noise level the second-order residual term matters.  Stop when the KKT residual
  < 1e-2 or an accepted step with ν ≤ μ changes the objective < 1e-5.  Gravity + DC synthetic: 14 steps,
  1.7 s; log10 σ −1.3 in the body (true −1), −2.00 outside (true −2); density 0.17 (true 0.3).
- `gl_data_scaling="auto"` (new default): the paper's max_ratio for potential fields, 1/std when a
  dataset is nonlinear.  Worker: any methods and model labels (≥ 2 models), `gl_cross_gradient`,
  `gl_gn_max_iter`, `gl_gn_tol`; results keyed by model and dataset label.

**DAG.**  `JointRegularizedInversionNode` runs the worker's joint paths (as RegularizedInversionNode does
for single ones): method one with any regularization and per-dataset overrides, or the group lasso
(`group_lasso={gl_*}`); outputs one PhysicalModel per model, per-model/per-dataset summaries and the
λ1 sweep as `selection`.  `JointInversionNode` is unchanged (saved workflows rebuild as before).

Tests: 356 passed, the 2 platform failures above (tests/test_joint_regularization.py 16,
tests/test_em_methods.py 7, tests/test_joint_node.py 3, test_group_lasso.py +12, test_data_pipeline.py
+2).  Not yet: the upload page has no controls for the per-dataset settings, model labels or
`gl_cross_gradient` (the pipeline accepts them), and MT/DC files are not read by the pipeline (Python
and DAG only); nothing of this has run on AWS.

**macOS launcher.**  `GeoInv3D.command` (repo root; double-click in Finder) does what GeoInv3D.bat does
on Windows: opens the page if the server already answers /api/health, else starts
`.venv/bin/python -m geoinv3d.api` on 127.0.0.1:8000 (`GEOINV3D_PORT`), waits up to 30 s and opens the
page.  Its Terminal window holds the server (closing it or Ctrl-C stops it), `caffeinate` keeps the
Mac awake while the server runs, and it says what is missing (no .venv, no cloud extras, no AWS
credentials).  Tested: start, second launch, window close (server and caffeinate gone), port taken.

---

## 2026-09-29 — MT / DC files in the pipeline; synthetic multiphysics run on EC2

**Pipeline.**  MT and DC datasets come in one .npz each (`_load_em_dataset`): DC `electrodes`
(n, 12) or `a`/`b`/`m`/`n`, MT `locations`, `frequencies`, `components`; `values`, optional `std`
(else noise_pct + a required noise_floor), `method_kwargs` (sigma_background, data_type).  The mesh
covers stations and electrodes; results plot one point per datum (DC: the electrodes' centre, MT: the
station); the viewer labels log-conductivity models and V / Ω·m / Ω data.  The single path now starts
from and regularizes towards the method's background (log σ_bg for MT/DC; it used 0, i.e. 1 S/m),
and the padding alarm measures the anomaly relative to it.

**Sensitivity threshold.**  Weights are clipped at 1e-12 of their maximum for potential fields
(SimPEG's default) and 1e-2 for MT / DC (SimPEG's DC examples; `SENSITIVITY_THRESHOLD`, also per
model in `JointSensitivityWeights`).  With 1e-12 the padding cells of a DC inversion ranged from
log10 σ −10 to +5.

**Synthetic data** (`examples/synthetic_multiphysics.py`, data in
examples/output/multiphysics_synthetic/): one block 200 × 200 m, 75–225 m deep, +0.3 g/cc, 0.03 SI,
0.1 S/m in 0.01 S/m; gz and TMI on 13 × 13 stations (modelled on a 25 m mesh), DC dipole-dipole
(a = 50 m, 7 lines, 238 data) and MT (3 × 3 stations, 100 and 1000 Hz, 72 data) modelled on the
inversion mesh itself (50 m core, 12 544 cells).  Why: on 50 m cells DC with 50 m dipoles is far off
(χ² of the true model 9044 for N = 170; 182 on 25 m cells; 2610 with 100 m dipoles), and cells fine
enough make MT slow.  One inversion then produced a resistive body from conductive-body data — the
modelling error, not the code.

**EC2 (ap-south-1, c5.xlarge; workspace "Synthetic multiphysics").**  χ² (N), mean in the body:
gravity sparse 178 (169), 0.092 g/cc; TMI sparse 201 (169), 0.020 SI; DC l2 255 (238), log10 σ −1.74;
MT l2 75 (72), −1.71 (true −1, background −2.00 exactly); joint gravity + TMI with per-model sparse
and bounds 188 / 212, 0.087 / 0.022 (like the separate inversions, as it should be without coupling);
group lasso + cross-gradient (λ3 = 0.1, λ1 from the L-curve corner) 0.277 g/cc and 0.030 SI — the true
amplitudes — with 58 % of |anomaly| in the body against 11–20 % for the others.  Its χ² (34 / 1259)
look uneven only because the jobs' errors did not match the synthetic noise (0.02 mGal for 0.0081,
1.5 nT for 3.3: a perfect fit gives ~27 and ~820); the script now uses the true errors.
The joint jobs with MT and DC were cancelled: MT with SimPEG's single-threaded SuperLU (no Pardiso or
MUMPS on the instances) takes ~30 s per forward run and ~40 s per Jacobian locally, slower on EC2, so
the group lasso needed ~27 min per λ1 point.  Speed-ups not done: pydiso (Pardiso + MKL; wheels exist
for Linux x86_64 / Python 3.13) or python-mumps on the instances, and reusing the accepted trial's
forward run at the next Gauss–Newton linearization (one of ~3 forward runs per step is repeated).
Scope from here: joint gravity + magnetics only.


## 2026-09-29 (evening) — Bouguer check, topography in the pipeline and the 3D view (finished in the next entry)

**Karnataka data (Desktop/karnataka_ap_gravity).**  The NGPM station table (ASCII/combined_NGPM_gravity.csv:
X/Y lon-lat, bouguer_an, elevation, observed_g, theoretical_g, toposheet; 22,816 stations, 1,941 in the
study window, 386–1,019 m) reproduces the Bouguer anomaly exactly as free-air minus a 2.67 g/cc slab with
GRS80 normal gravity (residual 0.63 mGal = the 1 mGal rounding; +0.74 mGal in both the roughest and the
flattest tenth of stations).  So: simple Bouguer anomaly, 2.67, NO terrain correction.  Rock samples
(Physical_properties_rock_samples.csv, 31 in the window): BIF 3.39 g/cc (+0.72 vs 2.67), amphibolite +0.25,
dolerite +0.31, gabbro +0.36, granites -0.03…-0.05 — above our +0.5 upper bound for BIF.  Drill/ is a
collar-only shapefile (448 holes, WGS84; 16 in the window, all iron ore at Kumaraswamy / Ramanadurga South,
12–125 m): no lithology or density.  No geology map is available.

**Done and tested (uncommitted).**  methods/bouguer.py: reduction density by least squares and a terrain-
correction verdict from the residual against the local relief (spread of station heights within 3 km);
rho refitted on the flatter half when a correction is found.  tests/test_bouguer.py (9 pass).  The upload
page runs the same check (bouguerCheck / bouguerTable / checkGravity in dag_interactive.html) under the
gravity card: amber "no terrain correction", green "applied", neutral for grids; verified headless on the
NGPM table, a synthetic corrected table and the NGPM GeoTIFF (JS numbers = Python).  The worker records
bouguer_check per gravity table in the result's datasets.

**Written but NOT yet tested (run the full suite first).**
- io/crs.py (new): looks_geographic, utm_crs, project (rasterio), working_crs.
- io/readers.read_station_table: text columns read as NaN, text-only columns dropped (the NGPM table's
  toposheet column used to make every row unreadable); Bouguer column names; _NOT_VALUE_NAMES.
- worker: _job_crs (params crs > first GeoTIFF CRS > UTM of lon/lat tables); _read_observations projects
  lon/lat tables before cropping and rejects grids in another CRS; _load_dataset(crs=);
  _load_topography(params, data_dir, crs, datasets): DEM in another CRS sampled in its own CRS, lon/lat
  point files projected, topography {"from_data": true} = surface from the data's station elevations;
  result["crs"], topography elevation_min/max, result["topography_grid"] -> topography.npz.
- viz/result_workflow: load topography.npz; ViewerGrid shows the relief layers (tensor: all top dz layers;
  octree: down to core depth below the lowest ground); model_3d.surface and model_3d.stations.
- dag_interactive.html: 3D "Terrain" (translucent surface with contours) and "Stations" layers with
  toggles, z axis "Elevation (m)" covering the ground; topography checkbox "Use the station elevations"
  (params topography {from_data: true}); the gravity note mentions it.

**Next.**
1. Page: project lon/lat tables to UTM in the browser too (GeoMesh parseTable/inspectFile and GeoPreview
   table(): the Area/Mesh steps still treat degrees as metres and warn); zone from a GeoTIFF's GeoKey 3072
   if present, else from the median longitude.
2. Tests: crs helpers, reader with text columns, projection + from_data topography through the pipeline,
   ViewerGrid relief layers, surface/stations in the workflow; headless check of the 3D terrain.
3. A local low-resolution Karnataka run: NGPM_BA.tiff as data + station elevations as topography, to see
   the stations at 386–1,019 m and the model under the terrain.
4. Then the geology-constrained model (agreed direction: a JSON spec, semi-automatic): units from the rock
   samples (editable), boreholes (iron -> BIF along the trace), manual interpreted bodies (polygon, depths,
   dip), no map for now; applied as reference model + per-cell bounds + smallness weights (a starting model
   alone does not change a linear, convex inversion); CDA by shifting data/bounds; raise the BIF bound.
5. Slack reply to the team: point 2 must say "no terrain correction" (the version in the chat is correct).
Git: pulled 7dcbe97 and 6ab5f41 (fast-forward); everything above is uncommitted.


## 2026-09-30 — Topography and CRS in the pipeline, terrain in 3D, geology constraints

**Coordinates and topography (tested).**  Each job works in one CRS: params `crs`, else the
first projected GeoTIFF's, else the UTM zone of its longitude/latitude tables (a geographic DEM
never becomes it; metre data with only a lon/lat DEM take the DEM centre's zone).  Lon/lat tables
are projected before cropping; the page does the same in the browser (GeoCRS: Snyder TM, the
zone from a GeoTIFF's GeoKey 3072) — the NGPM stations then share the grid's metre frame to
~100 m.  Both table readers keep rows with text fields (the NGPM toposheet column made them
unreadable).  Topography: a DEM in another CRS is sampled in its own; `{"from_data": true}` (page:
"Use the station elevations") builds the ground from the stations' elevations.  Results carry
`crs`, the ground's elevation range and topography.npz; the viewer keeps the relief layers
(tensor: all top dz layers; octree: to the core depth below the lowest ground), and the 3D view
has Terrain (translucent surface with contours) and Stations layers, an Elevation axis, the depth
slice starting under the lowest ground, and no render while the plot is a few pixels high
(Plotly's "axis scaling" error).  tests/test_topography.py (9).  Local Karnataka run at 2 km with
the NGPM grid + station-elevation terrain: EPSG:32643, stations at 392–994 m, 30,381 of 32,400
cells below the ground, 15 s.

**Geology constraints (methods/geology.py, io/vector.py, docs/geology_constraints.md).**  A JSON
spec: units (value = mean sample density − background, range = min…max ± margin, or given;
a smallness weight), placed by rock samples (column to a depth), boreholes (along the trace;
GSI field names by default), interpreted bodies (polygon/box between depths, dipping) and map
polygons (shapefile/GeoJSON, no GIS libraries needed).  Applied as reference model + per-cell
bounds + smallness weights (on SimPEG Smallness terms only) and the start at the reference;
the L1–L2 coordinate descent inverts d − G m_ref for the deviation (weights unused).  Single
gravity/magnetic inversions only.  Results: `geology` summary, reference_model.npy (viewer:
"Geology reference" layer), setting `geology` (its own branch in the tree).  Page: Inversion step
→ Geology constraints (drop the spec and its files; missing files block the submit; Review row).
tests/test_geology.py (13).  Karnataka at 2 km (examples/output/karnataka_gravity/geology/,
samples + iron holes, 68 constrained cells): χ²/N 1.131 vs 1.134, RMS 0.532 mGal both; the
constrained cells average +0.103 (reference +0.113) against −0.003 unconstrained — the data accept
density at the surface under the outcrops; the empty top kilometre was a regularization choice.
## 2026-09-30 (later) — Model builder on the upload page; constraint files optional

**Model builder (viz/dag_interactive.html, "Starting & reference model").**  The user asked for a
MARE2DEM-style way to build the starting model by hand, in 3D.  On the Inversion step: a map (the
data behind it) to draw polygons on (click vertices; double-click or the first vertex closes; Esc
cancels) or add boxes/cylinders; per body an outline, top/bottom below the ground, dip and dip
direction, a density (g/cc, sent as the contrast with the reduction density) or susceptibility (SI),
optional range (default ± 0.05 g/cc / ± 0.005 SI), weight and "Fixed" (tenth of the range, weight
≥ 100); a W–E/S–N section (Shift-click moves it; depth to 1.4 × the deepest body, mesh core bottom
marked); "Everywhere else" (reduction density, bounds).  The bodies become units + `body` sources of
the same spec (no backend change), merged after a dropped file spec.  Save model = the spec with a
`builder` field; dropping it back restores the bodies exactly.  Joint inversions show a note that
they do not use it yet.  Checked in the in-app browser on the NGPM grid: a drawn polygon (200–3000 m,
45° E, 3.1 g/cc) reaches the submitted job as Δρ +0.43 ± 0.05 in EPSG:32643 metres, and
`build_constraints` puts 2,979 cells per 500 m layer shifting 0.5 km east per layer; save → delete →
load restores it; the Karnataka file spec + a drawn body submit together (units BIF, mafic,
granitoid + the body; sources samples → boreholes → body).

**Constraint files optional.**  A spec whose samples or borehole files are missing no longer blocks
the submit: `build_constraints` skips a missing samples file (the units needing it are left out,
with bodies of those units), and a source whose file is missing (report `skipped: file not found`);
unknown units still raise.  tests/test_geology.py (15, incl. the builder's spec format).

**GitHub branch.**  `claude/cool-hawking-9zd59t` at 6ab5f41 (joint inversion with any
regularization, MT/DC in the pipeline, multiphysics example): every file of those two commits is
identical in this working copy, or has only this copy's later additions (CRS, topography, geology) —
nothing left to merge.

## 2026-09-30 (night) — Thin layers, resistivity and sharp boundaries in the starting model

The user asked to polish the model builder for thin layers with resistivity modelling (MT/DC)
in mind, and to borrow from ModEM and mtpy-v2.

**Backend (methods/geology.py).**  Bodies, maps and the new `layers` source give each cell the
share of its volume they cover (depths analytic; footprints sampled 4 x 4 where an outline
crosses them; tilted interfaces sampled where they cut a cell), and a later source takes its
share from what was there: a cell's reference, bounds and weight are volume averages, so layers
thinner than the cells are mixed in instead of lost (ten 30 m layers in 100 m cells: before,
7 of 10 vanished; now every one is kept and contrast x thickness is conserved down each column).
Samples and boreholes still mark whole cells.  `layers`: a stack everywhere or in an outline,
below the ground or by elevation, interfaces tilted about a pivot, the last layer optionally to
the bottom, `unit: null` = a free layer.  `property: "resistivity"` (ohm m in, log conductivity
inside, bounds swapped, samples geometric), with `mixing` log / conductance (MT) / resistance (DC).
`sharp` units: labels whose boundaries get face weights `sharp_factor` (0.01) on SimPEG's
first-order smoothness terms (ModEM's covariance "tears").  `"crs": "job"` marks metre
coordinates.  Worker: geology for single MT and DC jobs too (the method's background as the
free cells' reference), `smoothness_labels` packed with the task.  Tests: test_geology.py 26
(thin layers, half-space and outline, tilt, elevation, resistivity units/mixing/samples, face
weights, a DC pipeline run with a layered prior).  The DC Jacobian test's rtol 1e-5 -> 5e-5 (the
central difference's own error was 1.2e-5).

**Page.**  Parts are bodies or layer stacks; each body and layer holds a value (and range) per
property of the loaded data — density, susceptibility, resistivity — and each job gets the spec
of its own property (checked: gravity + DC "separate" gave the gravity job the box only and the
DC job the five-layer stack + the box).  Layer table with computed tops, reorder, weight, fixed,
sharp; "Paste a table" (`[name] thickness value [lowest highest]`, `-` = to the bottom).  Section
coloured by value (density contrast diverging, resistivity log, conductive red), free parts dark,
the mesh's vertical cells dotted, log-depth option, zoom; thin-layer warnings against the mesh's
dz.  Everywhere else per property (resistivity: background and mixing).  Save = the first
property's spec + `other_properties` + `builder` (version 2); save -> delete -> load restores the
stack and the body exactly.  The page's DC spec through build_constraints on 10 m cells: the 8 m
graphite layer mixed into its cell (2.15 ohm m with sand), the 0.05 degree tilt moving the stack
81 m down at the east edge — as drawn.

**From the repositories.**  mtpy-v2: log-increasing vertical cells (`z1_layer`,
`make_log_increasing_array`) — thin shallow cells are what resolve thin layers; the core cells
here are uniform (next step, not done); `assign_resistivity_from_surface_data` (surfaces from
grids; here planes, by volume).  ModEM: covariance mask codes (0 air, 9 ocean fixed, 1–8 free)
and tear rules — here `fixed` and `sharp`.

**Model step (later the same night).**  At the user's request the builder is its own wizard step:
Data → Area → Mesh → **Model** → Inversion → Review → Jobs (it needs the Area window for its map and
the Mesh step's vertical cell for the thin-layer checks, and comes before the regularization).  The
page widens to 1560 px for it: map 500 px high with the section under it on the left, the list and
form on the right (380 px), the layer table full width, then "Everywhere else" beside the file inputs.
Checked in the in-app browser: Area → Mesh → Model → Inversion → Review → mock submit → Back to Model;
the Review row and the submitted job carry the model.  Vertical grading of the mesh: not needed for
gravity/magnetics (a thin layer only counts as contrast x thickness, which the volume averaging
keeps); worth adding when MT/DC with thin shallow conductors is inverted for real.


## 2026-09-30 — The coupling of a joint inversion as a layer of its own

The user asked whether merging the group lasso and the cross-gradient into one method was
sound.  Literature (Colombo & Rovetta 2018; Haber & Holtzman Gazit 2013; Zhdanov et al. 2012;
Vatankhah et al. 2020; Utsugi 2025): a joint objective has data terms, a regularization per
model and a coupling, and the coupling is its own choice with its own assumption; the group
lasso is a coupling (joint sparsity, co-located support), not a per-model regularization, and
Utsugi's method has no cross-gradient.  So: split, and add SimPEG's couplings.

**Coupling layer (methods/coupling.py).**  `coupling`: cross_gradient, joint_total_variation,
linear_correspondence (two models; slope/intercept or coefficients), pgi (rock units), group_lasso
(its own ADMM solver), none; `resolve` maps the former keys (regularization_type "group_lasso",
cross_gradient_weight > 0).  JointInversion(coupling, coupling_weight, coupling_options); a generic
pair wrapper for two-model SimPEG terms (>2 models: every pair).  The group lasso's λ3 is kept as a
named hybrid, `group_lasso+cross_gradient`.

**Finding: the cross-gradient weight coupled nothing.**  For g/cc and SI models the cross-gradient
is ~1e-10 at the inverted models (quartic in the amplitudes); raw weights 1, 1e3, 1e6, 1e9 all gave
the uncoupled models (correlation 0.897–0.899 on the block synthetic of test_joint_regularization) —
the page's default weight 1 has been an uncoupled joint inversion.  `CouplingScale` (directive):
coupling off in iteration 1, then after iterations 1 and 3 each term's multiplier = w × λmax(reg
Hessian) / λmax(term Hessian); w = 1 weighs it like the regularization.  Correlation 0.90 → 0.93 /
0.97 / 0.99 at w = 0.1 / 1 / 10, both χ² near N.  `coupling_options.scale = "raw"` keeps the old.
JTV: each model scaled by 1 / rms(∇m_i) (unscaled, w = 10 left gravity at χ² 92 and magnetics at 0;
scaled, both fit at 0.1–10).  Linear correspondence ρ = 6χ: correlation 1.00.

**PGI (SimPEG PGI + its tutorial directives).**  Units {name, means, stds, proportion} per model
label; a background at the references is added with a tight spread, 2.5 % of the largest unit
contrast.  16x16x8 synthetic, 4x4x3 block (0.3 g/cc, 0.05 SI): L2 gives 0.045 in the block;
PGI with a 10 % background found 0 block cells (the smear counts as background), 1.7 % → 47/48 and
0.175, default 2.5 % → 46/48 and 0.246 (χ 0.041 of 0.05).  Learning the means (kappa 0) collapsed
them onto the smear (0.035) — off by default.  Result: pgi.membership and the final units.

**Plumbing.**  InversionTask joint_coupling / coupling_weight / coupling_options (packed; `coupling`
property); pipeline params coupling / coupling_weight (manual; auto 1) / coupling_options; results
`coupling` (+ multipliers), `pgi`, settings.coupling = first column of the workflow tree (PGI runs
skip the per-model columns); JointRegularizedInversionNode(coupling, …); dag_view labels.  Page,
Inversion step: "How the models are coupled" (coupling, weight in manual, the linear relation, PGI
units typed or filled from the Model step, descriptions with references); one "Joint — A + B"
entry per combination instead of a separate group-lasso entry; λ3 for the hybrid; coupling weight
sweep.  Checked in the browser: each coupling's submitted params and the Review row.
tests/test_coupling.py (13).  (pytest here needs stdin from /dev/null when run from Git Bash, or it
waits.)

**Coupling demonstration (2026-09-30).**  examples/output/coupling_comparison: three dense bodies (0.3 g/cc; χ 0.05 / 0.01 / 0),
six couplings with L1–L2 per model, blocks and dipping intrusions; report and two workflow viewers published as artifacts.
Fixes found on the way: JTV weighed by value once after iteration 1 with ε relative to the gradients (curvature matching gave
it ~0 weight against sparse models); PGI units named by nearest means (SimPEG reorders the mixture; names were swapped);
build_workflow takes a true model per property; runs in the tree are named by their coupling (weight in the coupling label).

**Why the group lasso wins (same day).**  A control (gl_coupling="none": the group lasso's ADMM solver, weighting, L2 and
L-curve, each model soft-thresholded alone) recovers 0.23-0.25 g/cc in the blocks against 0.25-0.27 coupled and 0.08-0.11 for
L1-L2 alone: most of the gain is the solver.  The coupling gives shared support: B's weak χ in 95 % of its dense cells (40 %
uncoupled; peak 0.027 vs 0.083 SI), a better magnetic fit (χ²/N 1.22 vs 1.36).  Cross-gradient: zero wherever either model
is flat - 443 of 611 cells with density structure have no χ structure (B weak, C none) - and it aligns directions, not the
amplitude/depth the separate inversions lose.  IRLS 20 -> 40 cycles: the uncoupled and JTV runs converge, densities +60-85 %;
cross-gradient and linear correspondence still hit 40.  Report updated (why section, control, IRLS table).

**Names and PDF (same day).**  The control is not a group lasso: nothing pairs the two values of a cell, so it is L1 + L2 per
value solved by ADMM.  Report, figure rows, coupling label and workflow names now read 1. L1–L2, IRLS, no coupling; 2. L1 + L2,
ADMM, no coupling; 3. group lasso, ADMM (the pairing is the coupling).  The report's cross-gradient section was dropped: with
40 cycles the cross-gradient gains 20–90 %.  make_report_pdf.py prints the report to report/coupling_comparison_report.pdf
(A4, headless Chrome/Edge; block layout in print, since Chrome overlapped grid rows across page breaks).


## 2026-09-30 (afternoon) — Karnataka with terrain: gravity again, magnetics, joint

**Inputs (examples/output/karnataka_inputs/prepare_inputs.py).**  Copernicus GLO-90 DEM (6 tiles,
Desktop/karnataka_ap_gravity/DEM; DEM − NGPM station height +1.5 ± 4.5 m).  Terrain correction at the
3,269 stations in and around the AOI (methods/terrain.py: polar integration over the DEM's bilinear
surface within ~0.5 km, 90 m prisms to 4.7 km, 450 m and 1.8 km vertical lines to 50 km, mean h and
mean h² in the coarse cells; ring plateaus within 3 % of the analytic hollow cylinder; flat prisms next
to the station counted the step to the neighbouring cell as a slab, 0.17 mGal on a 10 % slope).
Median 0.10 mGal, 77 stations > 1 mGal, max 6.7 on the Sandur ridge tops; gridded by thin-plate spline
and added to the NGPM grid at its 1 km nodes (the inverted anomaly changes by −0.7…+6.0 mGal).  TMI
(37.5 m grid, flown 80 m above the ground): 18 % of the variance is at wavelengths < 2 km, so the grid
is continued upwards by 920 m (methods/continuation.py) before sampling at 1 km (0.07 % left);
receivers 1 km above the 450 m-mean DEM.

**Pipeline.**  Stations inside the staircase of ground cells move to the top of their column
(`_lift_buried_stations`; result topography.stations_lifted; 1,900 of 5,040 gravity stations, 72 m on
average with 250 m layers); a dataset's own `station_height`; EC2: a just-launched instance that
describe_instances does not know yet is "launching", not "terminated" (2 of 8 parallel launches were
dropped and left running; terminated by hand); ec2_multi_run --parallel / --collect, waits at the vCPU
limit; scikit-learn on the instances (PGI).  Group lasso: `bounds` per model (sign constraints
projected before the group shrink = the exact proximal step, other bounds clipped after it); without
them the 2 km Karnataka test gave 21.9 g/cc in padding cells.  tests: test_terrain.py (16),
TestStationsAboveTheGround, TestBounds.

**Gravity with terrain (karnataka_gravity_terrain; 8 runs on c5.4xlarge, 4.5–8.5 min, $0.67).**
84×84×50 cells of 1 km × 250 m, 335,518 below the ground.  Main-body centroid below the ground: β 0.5 /
L1–L2 / 1 / 1.5 = 3.1 / 3.6 / 5.4 / 7.7 km (flat earth: 3.4 / 3.8 / 5.6 / 7.6): the depth follows the
regularization as before.  Steps at β = 1: 250 m layers 5.6 km, ground in the mesh 5.4, terrain
correction 5.4; the correction adds dense rock in the top kilometre under the ridges (up to +0.15
g/cc·km), a continuous dense line along the south-western ridge.  Integrated density vs flat earth
0.97–1.00.

**Magnetics (karnataka_magnetic; 13 runs, 12.6–15.4 min, $2.11).**  5 % + 10 nT, susceptibility 0–1 SI,
IGRF 2020 (42,100 nT, 19.3°, −1.4° grid).  Share of susceptibility × volume below the 10 km core:
β 0.5 / 1 / L1–L2 8–10 %, β 1.5 45 %, β 2 70 %, β 3 82 %, sensitivity weighting 63 %: the textbook β = 3
fills the bottom padding.  Sandur belt centroid 0.8 / 2.1 / 2.8 (L1–L2) / 2.9 / 3.7 / 5.6 km.  Every
model underfits the high south of the belt (> 150 nT at 57–113 nodes, < −150 nT at 0–3): remanence
likely.  Tests at β = 1: α_s 0.1, 500 m continuation and a flat earth give the same model at 5 km
(0.95–0.98); bound 0.3 SI the same product in 4× the volume; 2 % + 5 nT is reachable (RMS 16 nT);
without continuation χ²/N 1.45 and a smeared model.  Integrated susceptibility vs integrated density:
0.24 column by column; the magnetic sheets rim the dense belt.

**Joint inversion (karnataka_joint; 7 runs on c5.9xlarge / c5.18xlarge, $10.4; stopped early on the
user's request).**  Each model as in its single inversion (sparse, α_s 1, β 1, bounds); weight 1.
Uncoupled = the single inversions (0.97 / 0.94 cell by cell).  Cross-gradient: measure
Σ|∇ρ×∇χ|² / Σ|∇ρ|²|∇χ|² 0.43 → 0.07, but only 2 % of the susceptibility edges lie on a density edge
(30 % uncoupled): it is satisfied by keeping the edges apart; susceptibility becomes a smooth shell.
JTV: 44 % shared edges, gravity χ²/N 1.26 (target not reached in 44 it).  Linear correspondence
(ρ = 0.5 χ): density pulled up to 0–3.5 km, 46 % below the core, both fits 1.12.  PGI (iron formation /
greenstone / light granite, our values): stopped at it 43 of 60 (68 min, 1.4–1.6 min per it), fits
1.10 / 1.18, half of both models outside the core (it weights by sensitivity, not by depth).  Group
lasso (paper settings, with the new bounds): 28 min set-up + ~5 min per λ1 on 72 cores, stopped at λ1
9 of 13 (χ²/N 2.1 / 6.9); 96 % of magnetic cells dense too (35 % in the uncoupled control), 69 % of χ
below the core.  2 km study (local): group lasso without bounds −0.67…21.9 g/cc; with bounds gravity
0.40 / magnetics 2.86 (error weighting + discrepancy: 0.38 / 2.62), > 80 % of χ outside the core.

**Open (see the joint report, Section 5).**  Group lasso: per-dataset balance, a weighting that keeps
the model in the core (γ = 0.5 / 1 / 1.5 tests on the 2 km mesh were started and stopped; settings
in joint_params.py), a shorter λ1 path at full resolution.  PGI: run to the end, depth weighting,
measured units.  JTV: more iterations.  Magnetics: remanence (systematic residual south of the
Sandur belt).  Not committed: the workflow viewers (25–30 MB each; scripts/build_workflow.py).

**Another machine.**  examples/output/karnataka_inputs/README.md: the prepared inputs are in the
repository; `deploy/ec2_multi_run.py <case> --local --collect DIR` runs a case without AWS.
AWS this afternoon: about $17, all instances terminated (checked: no instance or volume left).
Tests: test_terrain, test_topography, test_data_pipeline, test_coupling, test_ec2_backend,
test_joint_* (131) and test_group_lasso (59) pass; the full suite was last run before these changes.
---

## 2026-09-30 (evening) — Exact bounds for the group lasso; scikit-learn for PGI (macOS session)

Pulled 9cb670c and 6ca6e51 (fast-forward); rebased onto b761832 / ba4ffe1 later.  On this Mac 2 of 419 tests failed: SimPEG's PGI
builds its WeightedGaussianMixture on scikit-learn and, without it, defines a stand-in with no
fit(); scikit-learn was in neither pyproject nor the EC2 package list, so PGI jobs on EC2 would
have failed the same way.  Added (pyproject, Batch image; on EC2 the 1.8.0 of b761832) with an explicit error in
gaussian_mixture(); 419 passed.

**Group lasso bounds** (`GroupLassoProblem(bounds=...)`, docs/group_lasso_joint.md "Bounds").
Written in parallel with b761832, whose s update projected the sign constraints before the group
shrink (exact) and clipped the other bounds after it (not exact within a group: e.g. ρ ≤ 0.35
with χ free), with λ1,max and the KKT check unaware of the bounds.  This version replaces it, with
the same `bounds=` argument and worker semantics; b761832's tests (TestBounds) pass unchanged.
Rather than a third ADMM block (ζ = t, t in the box: 2μ in the ζ systems,
new factors, and the cross-gradient normalization would have moved), the box goes into the s
update's proximal step, solved exactly: per cell 0, or clip(q′ r/(r + κ), lo, hi) at the root of
r = ‖s(r)‖ (bisection, vectorized).  ADMM, μ, factors, stopping and the λ3 normalization are
unchanged, and so are unbounded runs.  λ1,max and the KKT check are bound-aware; the physical
models are clipped after the scaling round trip (it passed an upper bound by 1 ulp).  Checked:
per-cell numerical minima (4 boxes), FISTA with the same step (objective to 10 digits, KKT 4e-7),
light-body data with ρ ≥ 0 (density stays empty, susceptibility keeps the body, λ1,max drops),
a box above the reference (no empty cells).  Worker: per model its first dataset's
bounds_lower / bounds_upper, else the job's; the pipeline applies the job bounds to every model
and says which per-dataset settings the group lasso does not use.  tests: TestExactBounds (6).
