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

**Upload page: regularization per model** (Inversion step, manual, joint jobs).  The worker
took per-dataset overrides (`datasets[i].regularization` → `joint_regularizations`) but the
page sent none, so both models of a joint job had one regularization and one pair of bounds —
e.g. no way to keep susceptibility ≥ 0 while the density contrast has both signs.  Each model
now has a row: regularization (As above / L2 / Lp + norms / L1–L2 + α / MGS, TV + focusing
percentile), α_s, lower and upper bound; empty fields send nothing.  The group lasso and PGI
rows have the bounds only; the page's property bounds now go to the group lasso too, and its
note no longer says it has no bounds (nor that joint L1–L2 is L2: it is IRLS).  Checked in the
browser (values kept when the coupling changes, errors for norms, percentile and a box inverted
against the job bounds, Review lists the rows) and by running the page's submitted parameters
through run_data_pipeline: gravity sparse IRLS, magnetics MGS with χ ≥ 0.

---

## 2026-09-30 (late evening) — The open problems of the Karnataka joint report (Section 5)

Merged the Karnataka commits (b761832, ba4ffe1) under the three local ones; the exact bounded
shrink replaced b761832's projection-then-clip (its TestBounds pass unchanged; the local class
is TestExactBounds).  Then each open problem of Section 5, first on the 2 km mesh
(examples/output/karnataka_joint/data/lowres_runs_fixed, measures of make_figures.py via
scripts/fixes_summary.py), then at full resolution on EC2 (data/ec2_runs_fixed).

**Group lasso leaves the core.**  gamma = 2 gives every column of X unit norm: every cell,
however deep or far out in the padding, is equally cheap.  gamma = 1 did not help (49 % / 83 %
of rho / chi outside the core, pipeline measure).  `cell_weights` / `gl_weighting="depth"`: each
cell weighed in the penalty as the SimPEG regularizations weigh it, volume x (z + z0)^-beta with
each model's depth exponent (beta = 1), w = 1/R scaled to unit columns on average (lambda2 and mu
keep their scale).  Report measure, 2 km: 35 % / 17 % of the models in the core -> 82 % / 62 %
(uncoupled SimPEG run: 88 % / 62 %).

**Group lasso fits gravity and magnetics unevenly.**  One lambda1 for both: along the sweep the
gravity data are fitted faster, and the L2 term (lambda2 = 0.3) caps the magnetic fit, so chi^2
= N in total is not reached (0.45 / 2.44 at the smallest lambda1); lambda2 = 0.03 reached it as
0.35 / 1.66 with ADMM at its iteration limit.  `gl_balance`: a data weight per model
(`GroupLassoProblem.scale_models`: into s_d and c_p, so X, its factors and mu stay; the group
norm and the L2 term keep their physical meaning through per-model weights g_p = 1/w_p and f_p
= 1/w_p^2, which the exact bounded shrink takes: s_p(r) = clip(mu q_p r / (a_p r + lambda1
g_p^2), lo, hi)), rounds of f_p = sqrt(chi2_p / N_p) with a secant for the total chi^2 = N,
warm-started.  2 km: two rounds, weights 0.60 / 1.68, chi^2 / N 0.96 / 1.02 (0.40 / 2.86 in the
report).  A first version scaled the whole model (data, L2 and group terms) and diverged on a
ridge-limited dataset (weights 1e-7 / 7e6): the L2 cap does not move when data and L2 scale
together.  Checked against FISTA on the weighted objective, and for equal factors against
lambda / f^2.

**Group lasso cost.**  Over-relaxation (alpha = 1.6, default now): 6,458 -> 4,438 ADMM
iterations on the 13-point sweep, same chi^2 to five digits; the fixed runs sweep 8 points
(lambda1 comes from chi^2 = N and the balance, not the corner).

**Joint total variation misses its target.**  DampedUpdateIRLS halved its beta step at every
reversal and never let it grow back: in the report's run the step fell to about 1/16 while
phi_d jumped about under the reweighting, then beta moved 1.7 % per iteration with phi_d 20-50 %
above the target until the 40 IRLS iterations were used up.  Now two corrections in a row in the
same direction make the step 1.5 x larger (up to SimPEG's).  Simulated with 8 % scatter and a
4 %-per-iteration drift: runs ending 30 % above the target 82 % -> 15 %
(tests/test_regularization.py::test_damped_irls_follows_a_drift).  2 km runs barely change (none
0.93/0.81 -> 0.90/0.90; JTV 0.92/0.97 -> 0.90/0.91).

**PGI is slow and not compact.**  Its cell weights were the tutorials' sensitivity weights; now
each model's own depth weights when its regularization asks for them (beta = 1 here).  2 km: 28
iterations in 83 s, 67 % / 51 % of the models in the core (58 % / 33 % before), but it ended
overfitted (chi^2 / N 0.57 / 0.67; 1.08 / 1.00 before): beta never moved from its start
(2.85e-12), because SimPEG's PGI schedule only lowers beta and the tutorials' beta0_ratio of
1e-2 already fitted the data.  beta0_ratio 0.1: 0.74 / 0.93 in 30 iterations; 1: 1.36 / 1.35
after 60.  Default with depth-weighted models now 0.1 (PGI_BETA0_DEPTH); the EC2 PGI run
launched with 1e-2 was terminated after 3 iterations and launched again.  At full resolution
0.1 still overfits (phi_d 0.42 N at iteration 13, falling), because SimPEG's schedule never
raises beta; a directive that raised it when every dataset was overfitted (and withdrew the
early stop) did not help reliably on the 2 km mesh (0.77 / 0.94 with beta0 0.1, 1.41 / 0.96
with 1e-2), so it was dropped, the run terminated, and PGI is not pursued further for now
(it needs tuning per dataset; its rock units are assumptions anyway).

**Remanence.**  MVI: `MagneticsMethod(magnetization="vector")`, `run_mvi_inversion` (SimPEG
Cartesian MVI, VectorAmplitude with the task's norms, depth weighting, |m_i| <= bounds_upper;
result: amplitude as the model, the (n, 3) vectors, the direction of the strong cells).  Page:
the magnetic card's Magnetization select; single inversions only (the joint paths refuse it).
tests/test_mvi.py: a block at I -30, D 120 under I 60, D 0 — induced chi^2/N > 3, MVI < 1.5,
direction within 30 degrees (-33, 122).  Karnataka 2 km: RMS 47 -> 26 nT; the 29 stations the
induced model underfits by > 150 nT (south of the Sandur belt): RMS 280 -> 123 nT, 5 of them still
off by > 150 nT (the signed mean, 260 -> 70 nT, first quoted here and in 7c7c7a3, hides residuals of
both signs); the strong cells point at I 74, D -49 (coherence 0.73) against the field's I 19, D -1.
Full resolution (EC2, 27 min): RMS 49 -> 28 nT; at the 92 stations off by > 150 nT RMS 321 -> 129
nT, 22 still off by > 150 nT (16 above, 6 below); I 74, D -79, coherence 0.75.

Page (Inversion step, group lasso, manual): cell weighting, its depth exponent and the balance;
the paper's settings remain selectable.  The pipeline's auto mode uses GROUP_LASSO_AUTO (depth
weighting beta = 1, errors as data scaling, chi^2 = N, balance).  docs/group_lasso_joint.md
"Field data", docs/joint_couplings.md (PGI weights), docs/magnetization_vector.md.  454 tests.

---

## 2026-09-30 (night) — Group lasso at full resolution: faster

The coupled full-resolution run of the second series took 124 min on c5.18xlarge (control 66):
py-spy on the instance showed the exact bounded shrink, single-threaded numpy bisecting all
2 x 335,518 values at every ADMM iteration once the balance had made the weights unequal
(about 0.7 s per iteration against 0.25 s).  Now: a numba kernel per cell in parallel
(NUMBA_SHRINK_MIN; the numpy path below it; they agree to 1e-14; 6.5 against 28 ms for the
Karnataka size on 15 cores); gl_data_weights to start the balance from a 2 km run's weights
(0.595 / 1.682 at 2 km, 0.625 / 1.600 at full resolution); gl_lambda1_selection "search": a
descent to lambda1_max / 10 and a secant search for chi^2 = N from gl_lambda1_ratio, no sweep
(the ratio carries over less well, 8.8e-4 against 2.0e-3).  2 km: sweep + balance 5,223 ADMM
iterations, 93 s; sweep from the weights 6,869, 115 s; search from the weights 2,711, 54 s, the
same models (82 / 62 % in the core, 74 % of the magnetic cells anomalous in density), chi^2 / N
0.85 / 1.08.  Expected at full resolution: about 40 min (20 of them the sensitivities).  The
problem counts ADMM iterations over all its solves (group_lasso.admm_iterations_total).
Page: the search among the lambda1 choices (with the ratio field).  joint_params:
group_lasso_fast (the 2 km start).  tests: TestSpeed.

---

## 2026-09-30 (late night) — Karnataka joint: the rock-sample constraints

The measured rock samples (density_report.md, magnetics_report.md; table in
examples/output/karnataka_joint/data/rock_properties/) put the belt's metabasalt and amphibolite
+0.28-0.32 g/cc above the 2.66 background, the lightest granite at -0.14 and the iron formation at
+0.73.  2 km, uncoupled and JTV (joint_params.VARIANTS, run_lowres.py --bounds KEY): density
bounds -0.15 / +0.30, +0.35, +0.40 stretch the dense body (half-max 2.5-8.0 -> 1.5-9.0 km) without
lifting it (centroid 5.1-5.6 km, <= 2 % within 1 km of the ground); beta 0.5 for the gravity model
starts it at the ground (0.0-5.5 km, centroid 2.7-2.9); beta 0.5 for the magnetics too cuts the
root of the steep magnetic bodies, so kept at 1.  chi^2 / N 0.84-1.14 for all: the data do not
choose.  The column mass under the main high falls 2.2 -> 1.6 g/cc km with the shallower body:
mass / contrast is not a depth-free thickness check.  Full resolution (EC2 c5.9xlarge x 2, 23 and
28 min, $1.29; data/ec2_runs_bounds): -0.15 / +0.35, beta 0.5 / 1: chi^2 / N 0.91-0.94, dense body
1.0-5.2 km at half-max under the main high (3.0-7.5 before), reaching the ground along the belt;
density correlates 0.42 with the unconstrained model, JTV vs none 0.98: the constraints move the
density more than any coupling.  JTV then merges the two steep magnetic bodies (668-678 km) into
one wider body of 0.2-0.5 SI (volume above half max 141 -> 83 km3, 15 % below the core).  The
~6 km of Maurya et al. 2023 is a conference abstract (full text not accessible). Report v2
Section 5 (constraint_figures.py, compare_bounds.py [--full]); EC2 case karnataka-joint
<coupling>_<variant>.

---

## 2026-10-01 — Workflow viewer: folding settings, titles per tree

The all-runs viewer (examples/output/karnataka_inputs/build_all_workflow.py) now holds the 1 km
runs only (37; --with-2km for the local studies) and, besides the user's gravity and magnetic
studies and the first joint series, the second series, the rock-sample constraints and MVI.
Tree page: column titles were pooled over all trees ("coupling / Regularization", "α_s /
Depth weighting": trees of different studies have different columns); now each tree has its
own row of titles, written by the builder (branch.title: Coupling, Regularization, α_s / L1
share / λ2, Depth weighting, Norms p, the SETTING_LABELS name otherwise), and each setting a
fixed column within its tree (branch.column), so an L1–L2 run skips the norms column.  "Hide
settings" folds the setting columns (the result cards, wider, still name every setting of a
run; remembered in localStorage).  A mesh shared by several trees is drawn once per tree
(dashed copies) instead of with edges across the page.  Scrolling moves the tree and stops at
its ends, ctrl/cmd + scroll zooms; +/- zoom about the centre; double-click a result to open it.

---

## 2026-10-01 — Karnataka reports: the published geology, a schematic map, a mineral assessment

The literature cross-check of three HTML reports made outside the repository (gravity Section 6,
magnetics Section 6, joint Section 7: references, the iron mines, gold occurrences and the
Mincheri copper block) is now produced by the builder scripts, from the shared module
karnataka_inputs/shared/literature.py (15 references, the localities in UTM 43N with the
Kumaraswamy B/C blocks from the IBM report, `cite`/`citet`/`references_html`, LIT_CSS; style.page
takes `extra_css`).  The distances from the localities to the strong columns are computed from
the models (they matched the hand-read ones; a few statements changed with the numbers, e.g. not
"Kumaraswamy on the strongest magnetic rock").  No published figure is copied: a schematic map is
drawn from the 450 m DEM (ridges > 80 m above a 12 km median, read as the iron-formation ridges;
the belt = their envelope, 42 x 16 km, strike 136°) and shown beside the models in each report.
With a 5 km edge band left out, 34 % of the strong magnetic columns lie within 1 km of a ridge
(11 % of the area) and 86 % of the dense columns of the constrained joint model inside the
outline (12 %).  The user's stratigraphic argument (iron formation at the top of the succession,
steep near-isoclinal folds, so a dense core with steep magnetic sheets on its flanks, the
two-flank reading of Mukhopadhyay & Matin rather than a simple synform) is in all three; in plan
it is clear in the dense rock, weaker in the magnetic (a magnetic body also in the centre).
New: examples/output/karnataka_minerals (make_figures.py, build_report.py): a mineral assessment
of the area from the three reports and the record: iron-formation horizons (141 km2; 100 km2
more than 3 km from the four located mines, segments F1-F7), the remanent zone south of
Kumaraswamy (46 km2), and why manganese, gold and copper are below what these data resolve.
The magnetic sheets are 1-5 km wide at 1 km cells; their susceptibility x width (median 0.54,
up to 2 SI·km) would need 3.6-13.5 SI in a single 150 m layer: several bands, and the thickness
of one is below the resolution (the 37.5 m survey grid, 80 m AGL, 300 m NE-SW lines across the
belt, could resolve them).

---

## 2026-10-01 — The upload page runs inversions on this machine; previews, estimates, live curves, re-runs, map layers

An end-to-end check of the upload page (drop a file, set up, submit, view) worked, but only
through a test harness: jobs could only go to AWS. Changes, as agreed with the user:

- **Local backend** (`geoinv3d/cloud/local.py`): a job is a child process
  (`python -m geoinv3d.cloud.local <job dir>`) running the same worker as EC2; jobs queue (one
  at a time by default, `--local-jobs N`), the server follows them through their files, so they
  outlive a restart; cancel, stop-and-keep and failure without an exit code are handled. The
  server routes each request by the job's `backend` (local, ec2, batch); `/api/health` reports
  this machine's cores and memory. On the page: *This computer* or *AWS* in the Compute card.
- **Previews**: *Preview first* runs the job here with cells ×2 (data thinned ×2); the
  full-resolution settings go with it (`preview_of`) and its card starts them, here or on AWS.
  The server now keeps every job's inputs (`~/.geoinv3d/inputs/<task>/`).
- **Estimates** (`/api/estimates`): time ≈ setup + rate × data × cells × iterations (×2 joint),
  rates per machine from finished jobs (defaults from the Karnataka EC2 runs: 8e-9 s on
  c5.4xlarge), memory against the machine's, cost on AWS; shown in the Compute card and Review.
- **Convergence** (`worker.assess_convergence`): a discrepancy-principle run converged when
  chi^2/N <= 1.1 before its iteration limit. "Converged: Yes" was set for every run that the
  user did not stop (the Karnataka auto run ended at 30 iterations with chi^2/N 1.41); now
  `result["convergence"]` says why not (iteration limit, IRLS limit) with advice, shown on the
  job card and the node, with a one-click run with twice the iterations. L-curve/GCV, the CDA
  path, the group lasso and PGI keep their own flags.
- **Live curves**: progress.json carries the iterations so far (`history`); the card draws φd
  against chi^2 = N and β while the job runs (local jobs report every 2 s).
- **Re-runs**: `/api/inversion/{id}/rerun` runs a job again on its kept files with changed
  parameters (quick fields for iterations, norms, weighting, bounds, mesh, data errors, or the
  whole JSON), from its card or its node; a changed regularization makes an auto job manual.
  On the test data: 2 km, 30 iterations χ²/N 1.41 → 1 km, 60 iterations, converged at 0.96.
- **Map layers**: GeoJSON/CSV points and outlines (degrees projected to UTM, km read as m) over
  the 3D view, the depth slice, the sections (where they cross) and the data-fit maps; kept per
  workspace (`/api/workspaces/{id}/layers`) or carried by a workflow (`map_layers`). The
  Karnataka viewers carry the mines, towns, belt outline and ridges
  (`literature.map_layers()`, `karnataka_inputs/shared/karnataka_map_layers.geojson`).
- **Page fixes**: the header counts follow at once when a job ends (the workspace refresh
  adapts: 4 s with local jobs running, 20 s with AWS jobs, 60 s idle, and stops for a deleted
  workspace); the texts no longer assume AWS; results show where they ran; narrow windows
  stack the side panels; a workflow laid out while its tab was hidden is redrawn when shown;
  the depth slice starts at the top layer that holds rock (it opened in the air above a flat
  earth); the re-run dialog is centred.

---

## 2026-10-02 — The upload page fetches the ground and the field; setups; checks before running

Ideas taken from an EDI → ModEM preprocessor (kashkoulimohammad/edi-to-modem-mtpy-v2:
automatic topography, a JSON configuration shared by its GUI and command line, hard checks
before writing the files), as agreed with the user:

- **Automatic DEM** (`geoinv3d/io/dem.py`, `POST /api/dem`): SRTM 1" tiles from the AWS
  terrain-tile archive (8 MB each, cached; a missing tile is sea), else ETOPO 2022 15" read by
  window from NOAA's tiled GeoTIFFs (named by their north-west corner, which the reference
  tool gets wrong), averaged onto a grid in the data's CRS (30 m × the factor that keeps
  3000 px), over the data plus 0.6 × the survey width (the padding a recommended mesh adds).
  For Karnataka: 6 tiles, 60 m, against the study's 450 m DEM a median difference of 3 m
  (r = 0.9987). The page fetches the file and uses it like a dropped DEM; a magnetic job on it
  ran as before.
- **IGRF-14** (`geoinv3d/methods/igrf.py`, `GET /api/igrf`): IAGA's coefficients (shipped in
  `geoinv3d/data/`), the igrf14syn geodetic conversion, Schmidt recursion; checked against
  finite differences of the potential with scipy's Legendre functions (< 0.5 nT) and the
  Karnataka study's field (42,094 nT, I 19.27°, D -0.93° true / -1.35° grid north, the study
  used 42,100, 19.3, -1.4 to grid north). The page computes it at the centre of the magnetic
  data on the survey date and fills the card; it needs the data's CRS (new field, detected
  from GeoTIFFs or the zone of lon/lat tables, typed for tables in metres; also sent to the
  worker as `crs`).
- **Setups**: *Save this setup* (Review) writes the jobs' params and the page's choices;
  *Start from* (Data) loads such a file or a bare params.json (applied once its files are
  dropped), or a finished job with its files from the server (`/inputs/{name}`; a preview
  starts from its full-resolution settings). Round trip checked: every manual setting and the
  thinning came back. A job's *Inputs (.zip)* (`/bundle`) has params.json, data/ and a README;
  run headless it reproduced the page's job (χ²/N 4.49).
- **Checks before running** (Review): data in the window, cells vs data spacing, core depth
  and padding vs the survey width (the shallow core behind the 51 % padding share), DEM
  coverage and relief, the inducing field (warns on the 50,000 nT vertical default), zero
  noise floors, few iterations, memory, AWS credentials, the CRS; fix buttons (recommended
  mesh, IGRF, get a DEM, run on AWS / here); errors disable the submit.
- The size check used the instance's memory for jobs on this computer; it now uses this
  computer's.

---

## 2026-10-02 — The data as interpreters see them: enhancement maps and wavenumber separations

After Yang et al. (2026, Ore Geology Reviews 198, 107581: a ground magnetic survey of the
Hongchuan Cu-Ni deposit, reduced to the pole, continued 10-70 m before its vertical
derivative, edge detectors with normalized vertical derivatives, a Butterworth regional,
then a sparse inversion), which the user asked to build into the page:

- `geoinv3d/methods/enhance.py`: gridding (lattices placed as they are, scattered stations
  interpolated, gaps kept), reflection padding with a cosine taper, upward continuation,
  vertical and horizontal derivatives, RTP (Blakely's Theta; damped below |I| = 30°),
  Butterworth and band-pass filters, THDR, analytic signal, tilt, theta, THDR of the tilt,
  NSTD and the positive normalized vertical derivative of a detector. Checked against
  analytic fields: the RTP of an induced dipole at I 60° / 45° matches the vertical-field
  one (r > 0.999, same peak; 0.98 damped at 20°), continuation within 0.4 %, the vertical
  derivative r = 0.99997, the tilt of a point mass crosses zero at sqrt(2) × depth.
- `methods/regional.py`: `upward`, `butterworth`, `bandpass` regionals, on the data gridded
  at their spacing and sampled back at the stations; the worker logs a label instead of an
  order. Upward continuation also damps the broad field (its regional off by half the
  broad field's spread on a test field, against 5 % for Butterworth): said on the page.
- `POST /api/enhance`: the files once (read as the worker reads them), then a data id; the
  requested maps, thinned for display. On the Karnataka magnetics (71 × 71 at 1 km) about
  1 s; at I = 19.3° the RTP is damped and the page says so.
- The Area step's "Look at the data": dataset, map (field, derivatives, edges, regional,
  residual), continuation height, NSTD window, the zero line of the vertical derivative, the
  map layers, and the regional controls mirrored from the data card (what the inversion
  uses). A Butterworth residual chosen there (23 km, std 276 → 193 nT) was inverted by the
  worker with the same numbers.

---

## 2026-10-02 — Inverting magnetic data reduced to the pole

Yang et al. (2026) invert the Butterworth residual of the reduced-to-pole anomaly, not the
total field. The workflow is meant for other areas too, so it is now an option:
`rtp: true` on a magnetic TMI dataset (the card's *Data inverted*). `worker._reduce_to_pole`
reads each file whole, reduces it on the data gridded at their spacing (methods/enhance.rtp,
damped below |I| = 30°), samples the result at the stations the job keeps (after window and
thinning), and sets the inducing field to [F, 90°, 0°] so that data and kernel agree; the
regional field is then removed from the reduced data. Refused with MVI (induced
magnetization is assumed) and for anything but TMI. Against a SimPEG forward model of a
block under a vertical field the reduction is exact to 0.01 % (r = 1.0000, peak 134.7 vs
134.8 nT). The page blocks it below |I| = 30° (Karnataka, I = 19.3°, is refused with a
button back to the total field) and warns below 45°; the enhancement panel shows the
regional and residual of the reduced data when that is what is inverted. A job at a test
field of I = 60° ran: reduced, Butterworth residual (std 294 → 205 nT), inverted under a
vertical field; result.json records the reduction under `datasets[i].rtp`.

---

## 2026-10-02 — A new area (Block-8, the whole survey): flight height, continuation before thinning, automatic field

The user started on Block-8, the whole aeromagnetic survey the Karnataka study's 70 km window
came from (TAIL_TMI_GE, 37.5 m, 5055 × 5944 nodes, 80 m AGL, UTM 43N). What the page lacked:

- **Height above the ground** per gravity/magnetic dataset (`station_height`): grid nodes have
  no elevation and were put on the ground.
- **Continuation before thinning** (`continue_to_m`): the study continued the grid to 1000 m
  before sampling it at 1 km (its "as flown" run aliased). `worker._full_grid_filters` reads
  each file at full resolution within the window plus max(10 km, 10 × the distance),
  continues it (and/or reduces it to the pole, in the same pass), samples it at the stations
  the job keeps; the stations then sit at `continue_to_m`. Against a forward model at the
  higher level r > 0.995. The checks warn when a grid is thinned to more than three times its
  height without it, with a button that sets the continuation to the spacing.
- **The inducing field from IGRF-14, computed by the worker** (`"igrf": {"date": ...}` on a
  magnetic dataset, the card's default): at the median of the job's stations, on the survey
  date or, when "Date not known", on 2020-01-01; the page previews it for the window's centre
  and shows the change over 2000-2025 (on the test window: 1,143 nT, 2.8° in I, 0.8° in D).
  For the Block-8 centre in 2020: 42,006 nT, I 18.49°, D -1.46° from grid north (I from 16.2°
  in the south-west to 20.7° in the north-east).
- The magnetic core-depth check stops at 25 km (the Curie depth bounds the sources) instead of
  a quarter of the survey width.
- **Map layers on the Area step's window map**, with the same list and "Add map layer" as the
  3D view (one workspace list), to help choose the window.

A 20 km piece of the real grid ran end to end: IGRF-14 for 2020 (date unknown), continued 920 m
on the 37.5 m grid, thinned to 1 km, inverted. Also: the user's workspace had been created on
the test server (the launcher found it running); it was copied into ~/.geoinv3d, and the test
server moved to another port.

## 2026-10-02 — The memory estimate counts the octree the job builds

On a 5 km window of the Block-8 grid (17,689 nodes at 37.5 m, 50 m × 25 m cells, 4 km core,
2 km padding) the Mesh step said ≈ 153 GB. The window was applied; the octree was not: the
page counted 0.9 × every core cell at the finest size (1.44 million), where a job's octree
keeps the finest cells in the top layers only (`refine_surface`, levels [4, 4, 4]) and has
107,845 below the ground.

- `POST /api/mesh/cells` builds the mesh the worker would (`_build_octree_mesh` or
  `_build_tensor_mesh`, flat ground) and counts its cells below the ground, cached; under a
  second for a 5 km window. Above 4 million cells in the finest layer it answers `too_large`
  without building (the whole Block-8 survey at 50 m).
- The page uses the count for the Mesh step's text and warnings, the time and memory
  estimates, the size check and the review; until it answers, or without a server,
  `GeoMesh.octreeCells` (12 cells per finest column; 9-15 measured on 5 km windows). The
  recommendation budgets octree cells when OcTree is chosen.
- That window now: ≈ 11.9 GB (7.6 GB of float32 sensitivities); thinned to 75 m (4,489 data),
  ≈ 3 GB.

## 2026-10-02 — The flight-height check says what a DEM does not give

With an SRTM DEM downloaded, the Review checks still said the magnetic grid "has no
elevations, so its nodes are put on the ground", which read as if the DEM should have fixed
it. A DEM gives the ground; grid nodes go to the ground plus "Height above ground", so
airborne data still need the terrain clearance (Block-8: 80 m). The check now says so, also
fires when only a continuation height is set (the continuation then starts from the ground),
and carries a number box and "Set" that fill the Magnetics card's field. Checks can carry
such a box (`fix = [label, action(value), {input}]`).

## 2026-10-02 — The ground checks look at the data's window, not the whole DEM

The user's DEM had been downloaded while the whole Block-8 survey was the area: ETOPO at 450 m
over 461 × 495 km, -44 to 1,817 m (the coast is in it). After choosing a 5 km window the
Review said "The relief (1,861 m) spans 74 vertical cells", the range of the whole file; within
the window that DEM gives 577-972 m, from 11 × 11 pixels.

- `GET /api/dem/{id}/relief?bounds=W,E,S,N`: the lowest and highest ground of a DEM the server
  made, within the bounds (voids left out).
- The checks report the ground within the data; the relief warning is only for tensor meshes
  (it adds layers from the lowest to the highest ground there; an octree follows the ground,
  ±8 % cells on the window measured), and says how many layers.
- A new warning when the DEM covers more than 16 times the mesh area or its pixels are more
  than twice the cells, with "Get a DEM for the area" (source back to automatic: SRTM for a
  window).

## 2026-10-02 — "Get a DEM" downloads for the window

The DEM button always took the whole survey's extent (Block-8: 190 × 223 km, plus 0.6 of its
width around it, 461 × 494 km): "auto" chose ETOPO (more than 16 SRTM tiles) and returned the
cached 450 m file, "SRTM" refused. So the new "DEM made for another area" check's fix did
nothing. It now takes the window chosen in the Area step (else the whole survey), with a
margin of the wider of 0.6 × its width + 2 km and the mesh's padding + 1 km: a 5 km window gets
SRTM 30 m over 15 km. The SRTM refusal says to choose a smaller area.

## 2026-10-02 — A DEM in longitude/latitude passes the page's checks

A DEM downloaded by hand (for example SRTM 30 m from ArcGIS) is usually in longitude/latitude.
The worker samples such a DEM in its own coordinates (checked on the Block-8 window: within a
metre of the UTM copy), but the page compared its degrees with the data's metres, said "does
not cover the data" as an error and disabled the submit. `topoExtent` now takes the box into
the data's UTM zone (the part inside on every side) and its pixel into metres; for data in
another projected CRS it warns that the extent cannot be checked in the browser.

## 2026-10-02 — Topography moved to the Area step

The user's suggestion: the DEM depends on the window, so choose it after the window. The
Topography section (drop a DEM, "Get a DEM", station elevations, flat ground) is now on the
Area step under the data density, its button names the area ("for this window" / "for the
whole survey"), and a downloaded DEM that no longer fits the window (moved out of it, or
made for more than 4 times the area) says to get it again. The checks' "Get a DEM" fixes go
to the Area step.

## 2026-10-02 — The 3D tab for octree jobs: from the ground, and a size the browser can draw

The user's 5 km job (OcTree, 37.5 m × 25 m cells, SRTM ground 571-1,020 m) showed nothing in
the 3D tab, and its slices reached 4,000 m into the air. `ViewerGrid` sampled octrees from
the top of the octree's box (5,194 m) down to the core depth below the lowest ground: 134 ×
132 × 313 = 5.5 million voxels, Plotly's isosurfaces over all of them. It now starts at the
top of the highest ground cell over the stations (1,019 m; flat ground: 0 m) and keeps at
most MAX_VIEWER_CELLS = 600,000 voxels, coarsening by whole cells, the finer of the
horizontal and vertical sizes first (tensor: block means; octree: sampled on the coarser
grid). The job: 67 × 66 × 89 voxels of 75 m × 50 m, workflow 4.4 MB.

Also: "View Full Results" only on nodes with results to show (an inversion's iterations,
model or data fit), not on mesh and survey nodes; and results record `l1l2_solver` and
`l1l2_weighting` (the node said "CDA" for the IRLS run; older results take it from their
regularization label).

## 2026-10-02 — Why the Block-8 5 km job did not converge, and checks for it

The user's job (5 km window, 17,955 data at 37.5 m, OcTree 37.5 m × 25 m, L1–L2 IRLS, bounds
[-0.2, 0.4] SI, no regional) ended at χ²/N 2.33 after 56 iterations, β cooled to 4e-23, 91 %
of the model "outside the core". The window lies on iron-formation ridges: -4,132 to +8,864 nT.
25,344 cells sat at 0.4 SI and 40,190 at -0.2 SI; the 8,864 nT peak was fitted at 4,758 nT.

Tests on the same window thinned to 75 m (75 m × 37.5 m cells, 4,556 data, 40-100 s each):

| run | χ²/N | outside core |
|---|---|---|
| as run, bounds [-0.2, 0.4] | 2.35 | 0.91 |
| trend surface removed, same bounds | 2.43 | 0.88 |
| bounds [0, 2] SI | **1.07, converged** | 0.72 |
| trend surface + bounds [0, 2] | 1.12 | 0.84 |
| trend surface + MVI (sparse) | 1.12 | 0.70 |
| 10 km window, trend + sparse (auto, unbounded) | 1.12 | 0.79 |

The bounds were the cause; a trend surface did not help. With [0, 2] SI, 1,643 cells still
sit at 2 SI. Most of the rest "outside the core" is in the lateral padding: the anomalies
run across the window's edges (the east edge is -4,132 to -647 nT), their sources continue
outside, and the metric weights by cell volume.

New Review checks for magnetic data, from the browser's previews of the window:
- an upper bound under 4 × (strongest anomaly / F) (fix: 2 SI);
- a negative lower bound (fix: 0);
- anomalies over 5 % of F (susceptibility above ~0.1 SI): self-demagnetization and remanence,
  MVI suggested;
- strong anomalies at the window's edges (more than half the data's 5-95 % spread from the
  median), with "Widen the window by 2 km".

## 2026-10-02 — Overfitting on the Block-8 window: a cross-validation, and a check on the error floor

The user's re-run (bounds 0-3 SI, 37.5 m cells, all 17,955 nodes, errors 5 % + 2 nT, L1–L2)
converged at χ²/N 0.93, after dipping to 0.32 at iteration 31, and its model looked over-fitted.
The error model gave the anomalies' zero crossings errors of 2-5 nT: χ²/N 3.9 for |d| < 100 nT,
0.6 for 1,000-3,000 nT.

Cross-validation: inverted every second node (4,556, at 75 m) and forward-modelled the model
at all 17,955 (SimPEG forward-only, checked against the run's own prediction to 0.2 nT):

| run | errors | regularization | RMS fitted / unseen | χ²/N fitted / unseen |
|---|---|---|---|---|
| A as re-run | 5 % + 2 nT | L1–L2 0.8 | 86 / 98 nT | 0.54 / 2.38 |
| B | 2 % + 90 nT | L1–L2 0.8 | 154 / 175 nT | 1.30 / 1.80 (not converged) |
| C | 2 % + 90 nT | sparse p = [0, 1, 1, 1] | 130 / 130 nT | 0.89 / 0.89 |

A over-fits: speckle 50 m below the ground and rings at 200 m; C shows the two NW-SE
iron-formation belts and fits unseen nodes as well as the others, but puts the strongest
cells 200-500 m deep (depth weighting and the compact norm).

New Review check: a magnetic noise floor under 0.4 % of the window's 5-95 % spread, with a
fix to 1.5 % of it (80 nT on this window) and 2 %.

## 2026-10-02 — The page says when the server runs older code

The user saw the old 3D grid again: their server had started at 11:47, the fix went in at
12:16, and Python is loaded once (its /api/workflow still gave 134 × 132 × 313 voxels from
4,394 m). /api/health now reports `code_changed` (a package .py file newer than at start),
and the page shows a banner asking to restart GeoInv3D.command. The page itself is read on
every request, so it is always the new one.

## 2026-10-02 — Automatic data errors

`noise_pct` / `noise_floor` can be "auto" (worker.auto_noise): 2 % of each datum and a floor of
1.5 % of the 5-95 % spread of the data inverted, after the window, thinning and the regional
(a number given for one of the two is kept). The result records `noise_auto` with the spread.
The rule is the one the Block-8 cross-validation supported (2 % + 90 nT: χ²/N 0.89 on the
nodes fitted and on those left out, against 0.54 / 2.38 with 5 % + 2 nT); it is a starting
point, not a calibration: the misfit a model can reach depends on the model too
(remanence, bounds, cell size).

On the page the gravity and magnetic cards have "Automatic", ticked by default: the fields
are disabled, the card previews the floor from the window (Block-8 window: ≈ 85 nT; the job
from the page got 84.2 nT), the Review lists the errors, the floor checks skip automatic
errors, and setups, "start from a job" and the re-run dialog keep "auto".

## 2026-10-02 — The workspace's workflow cache is rebuilt after an update

After the restart the user's old jobs still showed the old 3D grids: the workspace workflow
is cached (~/.geoinv3d/workspaces/cache/<id>.json, 98 MB here) under a key of the finished
job ids only, and a job that finished at 13:54 had made the old server rebuild it with the
old ViewerGrid just before the restart. The key now includes CODE_MTIME_AT_START, so a server
with newer code rebuilds it (and the page, polling the key, reloads it).

## 2026-10-02 — Deleting a run

`DELETE /api/jobs/{id}` deletes a job that has ended: its record, and the files this server
keeps for it (inputs kept for re-runs, the local run's folder or the result fetched from
AWS, the temporary copies), and takes it out of every workspace. A queued or running job is
refused (409: stop it first; `DELETE /api/inversion/{id}` stays the stop). Saved comparisons
hold their own copy of the workflow and are not touched; a task id that is not a plain name
is not followed into the file system. The page has "🗑 Delete" on the cards of ended jobs and
"🗑 Delete this run" on a run's node, after a confirmation; the workflow and the list are
read again.

## 2026-10-02 — MVI in the memory estimate; the L1–L2 weighting is the CDA's; the padding note

- The user's MVI job (17,822 data, 143,237 cells, 40 m × 25 m) swapped for 8 minutes without
  an iteration (19.4 of 20.5 GB swap used, 19 % CPU): MVI has three components per cell, so
  G was 30.6 GB, and the page had counted one. The size check, the time estimate and the
  Mesh step now count 3 for MVI (that setup: ≈ 50 GB, "not enough memory"), and follow the
  Magnetization choice.
- l1l2_weighting (wS1 / wS2) and the λ range are used by the coordinate descent only; SimPEG
  IRLS weights cells by √diag(JᵀJ), i.e. wS1. Both are grayed out under IRLS; the note no
  longer calls α → 0 "smooth" (the L2 part is on the values).
- The padding note says where the model is: beside the data area (anomalies across the
  edges: widen the window) or below the core (deeper core, bounds), and suggests a compact
  regularization only to runs without one.

## 2026-10-02 — Compare with: another run in the same 3D view

The 3D view's side panel lists the other runs of the workflow shown (the workspace's, or a
comparison of chosen jobs). The one chosen is drawn with the current run: its shells on its
own grid (meshes and windows may differ; Plotly takes x, y, z per trace) in the second colour
map (green / purple) with its own colour bar and range, the scene's extent the union of the
two; and on the depth slice and the sections the outline of its bodies (|value| above the
same threshold share), sampled at the current run's cells. A same-run overlay of another
property gives way to it; a true model stays.

Also: /api/workflow refused an unknown id only after _refresh had stored an empty "FAILED"
record of it; it now looks the id up first.

## 2026-10-05 — Slices at the mesh's resolution, and a profile across the strike

The 3D tab's depth slice and sections were drawn from its grid, coarsened for the browser
(40 m cells over 5 km: 80 m). `viz/sections.py` samples a result's own mesh at half its
finest cells (at most 400 samples a side, 300 down): `ResultModel.plan(level, ref)`,
`.line(a, b, ref)` with ``ref`` "ground" (depth below the ground, the ground at 0) or
"elev" (cells above the ground blank), and `.strike_profile()`: the principal axis of the
strongest cells on a depth slice, the profile across it through their centre (the user's
window: strike 140°, the NW-SE iron-formation belt). `GET /api/inversion/{id}/section`
serves them (the last six models kept in memory; a 296 × 200 section, 0.6 MB, in 0.01 s).

The page draws every slice with one renderer: an image at the samples' resolution,
smoothed; turbo from 0 to the 99th percentile for models without negative values, the
diverging map for signed ones; its own colour bar; equal scales in plan; the ground line on
elevation sections; map layers (on any line, via a frame); the compared run's outline at
the same resolution. A new Profile A-B panel follows the strike by default, any line
dragged on the depth slice replaces it ("back across the strike" restores it), and its
plane is outlined in 3D. "Vertical axis" switches depth below the ground / elevation. For a
workflow file the same slices are sampled from the 3D grid.

## 2026-10-05 — The robustness check

The user found the 5 km window's models changing with the regularization and the depth
weighting. Their six runs (all χ²/N 0.90-1.09) agreed on where the bodies are (a NW-SE belt,
a body in ≥ 5 of 6 runs over 16 % of the window at 300 m) but not on their depth (the
moment-weighted centre of the bodies 290-450 m below the ground for L1–L2, 1,250-1,360 m for
smooth L2 and MVI): the depth is the settings', not the data's.

`POST /api/inversion/{id}/robustness` runs ROBUSTNESS_VARIANTS on the job's inputs, a group
of jobs marked `robustness_of` (on this computer unless asked; automatic errors; thinned to
twice the spacing above 6,000 data; MVI skips L1–L2). `GET .../robustness` lists them and,
from two finished, `ensemble_summary`: the share of the area with a body in ≥ 80 % of the
runs at 100, 300, 600 and 1,000 m, the columns with one in the top km, each run's bodies'
centre depth; `grid=1` adds the agreement on the job's 3D grid; `section?field=agreement`
slices it. A body is above 25 % of the run's 98th percentile sampled evenly through the core
(cell by cell an octree's many small surface cells raised compact runs' thresholds 2-3
times). The node has "🧪 Check robustness" and then the runs and the summary; the 3D view
draws the robust bodies (gold) and colours slices by the agreement on request. Tested on a
small run: six runs in about two minutes, all χ²/N 0.98-1.08.

## 2026-10-05 — The robustness check shows its progress; profiles drawn by clicking

The user started a check and saw no change: the runs were queued on this computer (six
after six, about 1.5 min each at 75 m), but the node kept "Starting…" (it refilled the
selected node, which could be another). It now refills the run's own node at once and every
15 s: how many runs are left, about how long (the median of those finished,
`minutes_per_run`), and the group's name on the Jobs page. The profile A-B has a ✎ Draw
button: click A, then B on the depth slice (the line follows the pointer; Esc cancels; a drag
still works), the depth slice saying what to do.

## 2026-10-05 — The agreement from any run of a robustness check

"Colour slices by: agreement" stayed disabled for the user: the check's six runs are nodes
of the workflow too, and only the checked run's node had the agreement. `robustness`, and
`section?field=agreement`, now resolve a run of a check to the run it checks
(`_robust_base`) and compute the agreement on the requested run's own grid; the node says
"this run is one of the check of …". When the agreement is unavailable the 3D panel says
why (no check of this run; fewer than two runs finished; not a run of the server).

## 2026-10-05 — A Bayesian posterior instead of the robustness check

The user found the robustness check (six chosen settings) unconvincing and asked for a
Bayesian inversion. Removed the check (its routes, agreement code and page parts); the jobs it
made stay as ordinary jobs.

`methods/bayes.py`: magnetic (induced) and gravity data are linear, so with Gaussian errors
and the smooth L2 regularization as a Gaussian prior, N(m_ref, (β P)⁻¹) with
P = Σ a_i B_iᵀB_i (SimPEG's terms with their depth / sensitivity weights), the posterior is
exact in data space: Q = P⁻¹Gᵀ (one SuperLU factorization, N solves), the eigenvalues of
Wd G Q Wd, then the mean for any β, χ²(β) and the evidence log p(d | β, s²) in closed form,
and RML samples (data + an error draw, reference + a prior draw) at two products each.
Checked against the normal equations (float32: 1e-4), the exact posterior covariance (MC
std / exact 1.00), the Gaussian density (evidence) and a known error scale (s 0.5 found).

`worker.run_bayesian_inversion` (`regularization_type: "bayes"`, `bayes_samples`,
`bayes_beta` "discrepancy" | "evidence", `bayes_threshold`): mean, posterior std, prior
std, the probability of a body (above a quarter of the mean's 98th percentile under the
stations, volume-weighted), the samples; `POST /api/inversion/{id}/bayes` starts it from a
finished run (auto errors, thinning to 2× above 6,000 data, refused when G and Q would not
fit: 16 bytes per entry + 2 GB, measured 12.2 GB for 4,489 × 141,759).

On the Block-8 5 km window (75 m data, 40 m cells, 259 s): β for χ² = N 1.3e-3; the
evidence would take β 1e-4 and the errors ×0.33, fitting the gridded data to χ²/N 0.014 —
it assumes independent errors, which a grid's interpolated values are not — so χ² = N is
the default and the evidence is reported. Either way: the variance reduction is 0.4-10 %
per cell (the data do not fix single cells); along the profile across the strike the
bodies' top is at 6 / 19-44 / 70-130 m (10/50/90 %), their base anywhere from about 0.5 km
to below the core (2.5 km).

## 2026-10-05 — Deleting a group of jobs at once

A group on the Jobs page (a sweep, or the twelve jobs the removed robustness check left in
the user's app) has "🗑 Delete the group" once none of its jobs runs: one confirmation, then
`DELETE /api/jobs/{id}` for each, a toast with how many went (and why any did not).

## 2026-10-05 — A posterior after its run; a clean workspace

The user found the workspace's workflow cluttered: a Bayesian posterior showed as one more
run (its thinned data made it a tree of its own). `build_workflow` now draws a run whose
`_posterior_of` (the job's `bayes_of`) is in the workflow as a `BayesianPosteriorNode` after
that run's node, on its own grid, with no mesh, survey or branch nodes (alone it stays a
run). The page draws it as a gold dashed card in an "Uncertainty" column after the results,
level with its run (tree layout), or below it (layers). A posterior started from a node joins
the workspace being viewed. `POST /api/workspaces/{id}/jobs` adds existing jobs to a
workspace (a job may be in several), with their posteriors.

## 2026-10-05 — A compact prior for the Bayesian posterior; the threshold on request; one colour range

The user found the smooth-prior posterior useless for targeting: "P ≥ 0.8" filled the
subsurface. On the Block-8 window (local-1b0f5945128a) the posterior std (0.23 in bodies,
0.20 outside) exceeded the body threshold (0.126 SI): over the core volume (under the
stations, 0-2.5 km) 19.7 % had P ≥ 0.8 and 25.9 % 0.2 < P < 0.8, growing with depth; the
data reduced the prior variance by 1 % in the median cell. The smooth prior is honest, and
says the data alone fix little; it is no map of targets.

`bayes_prior` "compact" (the default; "smooth" keeps the old one): the most probable model
is the Lp (0, 1, 1, 1) inversion, magnetics 0 to `BAYES_MAG_UPPER` = 1 SI unless an upper
bound is given; the samples are the Gaussian (Laplace) approximation around it from its
last IRLS weights (`LinearGaussian.perturbations`: RML with zero data and zero reference,
added to the mode, cut at the bounds, per cell where geology constraints give them), with
the IRLS eps floored at `BAYES_EPS_SHARE` = 0.25 of the body threshold (per core cell
size for the gradients): with eps → 0 a p = 0 weight is (m_max / eps)² larger in the
empty cells, which could not move at all. The body threshold is now half the model's
volume-weighted 98th percentile (a quarter before), and any other one on request:
`ResultModel.probability(threshold)` from the samples, `threshold` on `/section`,
`GET /api/inversion/{id}/probability?threshold=` on the 3D grid, and a box under
"Probably a body" in the 3D view. Upload → Regularization → "Bayesian" runs one directly
(prior, samples, threshold; memory 16 bytes × data × cells + 2 GB; single gravity or
induced magnetic data).

The same window (4,489 data × 141,759 cells, about 5 min, 12 GB), core volume:

| prior | P ≥ 0.8 | 0.2-0.8 | std in bodies / outside | |
|---|---|---|---|---|
| smooth (the user's) | 19.7 % | 25.9 % | 0.23 / 0.20 | variance −1 % |
| compact, no cap | 2.7 % | 0.5 % | 0.74 / 0.05 | bodies at 300-1,200 m, up to 13.7 SI |
| compact, ≤ 1 SI, no eps floor | 14.6 % | 1.4 % | 0.12 / 0.04 | |
| compact, ≤ 1 SI, eps floor (default) | 14.6 % | 1.4 % | 0.14 / 0.05 | variance −63 % |

Along the profile across the strike the tops are at 19, 106-131 and 294-394 m (10-90 %).
Without a cap the positive compact model packs the anomaly into a few cells of 10+ SI deep
down; capped at 1 SI it needs 15 % of the core volume at the cap, from about 100 m to 2 km.
Either way the induced, positive model strains: at I 19° (IGRF, F 42,085 nT) an induced
anomaly is mostly a low, but here the high (+8,845 nT) is twice the low (−4,109 nT, 1.65 km
east and 0.75 km south of it), which points to remanence (or self-demagnetization in
high-susceptibility BIF). The user's own runs say the same: L1-L2 needs −9.5 to +6.7 SI,
Lp (0, 1, 1, 1) with sensitivity weights up to 195 SI. The posterior is conditional on its
prior; a narrow one is the prior's narrowness as much as the data's.

The 3D view took its colour range from the largest voxel, the slices from the 1st-99th
percentile: on local-e43be0cd791d (Lp 0, 1, 1, 1; 99 % of the voxels below 0.06 SI, the
largest 40 SI and 195 SI in the padding) the profile showed every cell above 0.024 SI
saturated, the 3D view (20 % of 40 SI) only those above 8 SI, so the deep cells the profile
showed were not drawn. `model_3d.range` (`result_workflow.robust_range`, which the slices
use too) is now the 3D view's range as well; beyond it the colours saturate. A posterior's
"Open the probability in 3D" opened the previous result (`data-bayes-3d` is
`dataset["bayes-3d"]`, not `bayes3d`), and its card's status never refreshed (a node
compared with an id).

## 2026-10-05 — One colour scale; shells that peel; slices that follow a drag

The user still saw the slices and the 3D view coloured differently, the bodies changing
shape with the 3D threshold, and dark half-drawn slices while dragging a section.

Colours: a model without negative values was turbo from 0 to its 99th percentile in the
slices, but blue-yellow-red from its 1st to 99th percentile in 3D; and the two decided
"signed" differently (the slices on the minimum, the 3D view on the range).
`result_workflow.model_signed` (both signs present) is now the one rule, `model_3d` carries
`signed` and the slices' `range`, and the page's `modelScale(range, signed)` colours the 3D
shells, its colour bar and every slice: blue to red about 0 for signed models, else turbo
from 0 (from the lowest value when that is far from 0) to the 99th percentile.

Shells: each sign's three shells were spread evenly between the threshold and 90 % of the
peak, so every one moved with the threshold. They now sit at fixed shares of the colour
bar's |max| (`M3D_LEVELS` 20, 40, 60, 80 %) plus one at the threshold; a higher threshold
moves only the outermost inwards and drops the fixed ones it passes (thresholds 20 / 30 /
45 / 70 % on local-e43be0cd791d: shells at 20-40-60-80, 30-40-60-80, 45-60-80, 70-80 %).
Each shell's mesh is kept per grid and level, so a new threshold computes only its own
(270-520 ms for 75k-250k triangles).

Slices: a slice was asked for only after a 140 ms pause, so during a drag the page drew
the stand-in from the 3D grid: coarser, coloured by its extremes, and under the highest
ground reaching the relief deeper than the server's (the grid's bottom, not the core
depth), where it is empty, hence the dark band. Now one request per slice is out at a time
and the newest position goes next (a slice of the 5 km window: 11-21 ms on the server,
0.3-0.6 MB); while it comes the slice's previous fine one is shown; the stand-in (first
draw only) has the server's colours and depth (`model_3d.depth_max`) and samples the
probability when that is shown (nothing for the posterior std, which only the server has).
A slider redraws only the slices it moves (the E-W slider: that section and the depth
slice's line), once per frame. A 25-step drag of the E-W slider: 25 requests, one stand-in.

## 2026-10-05 — Alt Carbon's look: white and purple, the logo

The page was dark (#0e131a, an orange accent). It now follows Alt Carbon's site: white
surfaces, a very light lavender for panels and inputs (#f8f6fd, #f0ecfb), near-black text
(#1e1b2e) and the brand purple #7b4cff (sampled from altcarbon.com) as the accent, with a
3 px purple strip over the header, the Alt Carbon logo next to "GeoInv3D", and Upload as a
purple pill like the site's "Remove CO₂". The CSS variables on `:root` carry the theme;
canvases and Plotly read the same colours from `TH` (they cannot use CSS variables).
Warnings, which used the old orange accent, have their own amber (`--warn` #b45309); the
light reds, blues and teals of messages are their darker shades; the slices' line colours
(depth slice, E-W, N-S, profile) are darker versions of the same hues; stations in 3D and
the workflow's arrows are a soft grey-lavender instead of cream and dark grey.

The logo is `geoinv3d/viz/assets/altcarbon-logo-black-horizontal.png` (package data),
served at `/assets/{name}` (plain names only) and written into stand-alone viewers as a
data URI (`serve_dag.inline_assets`), so it shows without the server too.

## 2026-10-05 — No sources below a depth; showing down to a depth; one colour map

The user knows the deep anomalies of the Block-8 runs are not real and does not want them
shown. Two levels:

* `max_source_depth_m` (Upload → Mesh "Max source depth", ↻ Run again with changes, any
  saved setup): gravity and magnetics only; `run_data_pipeline` takes the cells whose
  centre is deeper than that below the ground out of the active set, like the air, so they
  stay 0 and the data are fitted by the cells above (refused for DC / MT, where the cells
  taken out would be air, and when no cell is left). The result's settings record it, so
  the workflow tree splits runs with and without it ("sources above (m)"). Hiding a wrong
  deep body is not enough: while it is in the model it carries part of the fit and bends
  the shallow part too; with the depth limit the misfit says whether the data can do
  without it (if not, the depth is too shallow).
* "Show down to … m below the ground" in the 3D view: only the display. The shells (values
  deeper set to 0 on the 3D grid, per grid and depth), the box and the slice outlines stop
  there, sections ask the server for `depth_max` (elevation sections are blanked below it),
  and a depth slice below it says so. Kept in the browser (localStorage).

Colours: unsigned models were turbo (blue-green-yellow-red, so "green to red" on a mostly
weak model), signed ones blue-yellow-red. Every model now uses the same blue-yellow-red map
about 0, so 0 is the same pale colour in every run: both signs the whole bar, positive
models its red half from 0, negative ones the blue half; only values far from 0 on one side
(a log resistivity) spread the map over their own range. Turbo is gone. (A compared run
keeps its purple-green map, to tell it apart in the overlay.)

## 2026-10-05 — The threshold as a share of the largest value, as before

The user liked that the old 3D view did not show the deep anomaly of local-e43be0cd791d.
It did so only because its threshold was 20 % of the largest voxel (40 SI): it drew the
305 voxels above 8 SI (98 % of them 300-1,000 m deep) and nothing weaker, so the weak deep
cloud the slices showed (0.01-0.06 SI, 80 % of it below 1 km) was left out with every other
weak body. That is kept as a choice under the threshold, "% of the largest value (only the
strongest)", next to the default "% of the colour bar's max (as the slices)": the threshold
and the fixed shells are then shares of the largest |value| shown (`m3dBase`), the colours
stay the shared scale (saturated), and the colour bars no longer blank inside a threshold
beyond them. The threshold's value is shown next to the slider ("20% = 8").

## 2026-10-05 — One threshold again; a note on extreme cells

The user found the two threshold bases (colour bar / largest value) one too many, and wanted
neither the faint values nor the extreme ones. In their workspace the extreme ones belong to
one run: node 14, local-e43be0cd791d (MVI, Lp 0, 1, 1, 1, sensitivity weights, no bounds):
up to 195 on the mesh (92 % of its anomaly in the padding, its own warning) and 40 in the 3D
grid, 1,672 times the top of its colour bar (its 99th percentile, 0.024). Node 10,
local-52e170aaf9d2, is an L1-L2 re-run with bounds [0, 3]: nothing above 3, and over 1 % of
its cells at the cap. Hiding the extreme cells cannot be done by a threshold: on node 14
the cells above 10 times the colour bar's top are joined to everything shown at the default
threshold (labelled bodies: 100 % of the shown volume would go), and cutting them alone
leaves hollows; every other run has none (largest value 1-2.4 times the colour bar's top).
So the threshold is a share of the colour bar's max again (the select is gone, its value
shown as "20% = 0.38"), and a run whose largest value is more than 10 times the top of its
colour bar says so under the colour bar: those cells are drawn in its darkest colour, do
not set the scale, and point to a re-run with an upper bound (the MVI bounds each
component) or a max source depth.

## 2026-10-05 — The OcTree's refinement under the ground, chosen on the page

The OcTree refined only 4 layers of the finest cells under the ground (`refine_surface`
with `padding_cells_by_level` [4, 4, 4]: on the Block-8 window, 40 x 25 m cells to 100 m,
80 x 50 m to 300 m, 160 x 100 m to 700 m), which the user found coarse. Upload → Mesh now
has "Finest-cell layers", "2× cell layers" and "4× cell layers" (OcTree only), the depths
they reach, the cells counted by the server on the mesh the job builds (`/api/mesh/cells`
with `octree_levels`) and the job's memory (sensitivities × 1.5 + 0.5 GB) against the
computer's. `run_data_pipeline` checks them (whole numbers 0-64, the first at least 1) and
records them in `mesh_design.used`, so runs with different refinement get their own mesh
node; ↻ Run again with changes takes them too. On the 5 km window at 40 m (flat ground;
the DEM adds about 15 %): 152,413 cells for 4/4/4, 237,874 for 6/6/6, 319,171 for 8/8/8
(40 m cells to 200 m, 80 m to 600 m, 160 m to 1,400 m; 9.1 GB with 4,489 magnetic data),
444,002 for 10/10/10, 576,233 for 12/12/12. Before the server answers, the page estimates
2.38 a + 0.57 (b + c) − 5.7 cells per finest column (within 16 % of those counts).

## 2026-10-05 — Block 8 with the rock-sample bounds; MVI as a survey of its own

Runs on the 5 km window (OcTree 40 x 25 m, refinement 8/8/8, sparse 0, 2, 2, 1, sensitivity
weights; gravity: the 8 NGPM stations in the window, complete Bouguer anomaly, a plane removed,
0.5 mGal; magnetics as before, 75 m), bounds from measured_by_rock_type.csv (block: density
2.60-3.40 g/cc, so -0.07 to +0.73 g/cc against the 2.67 reduction density; susceptibility
0 to 0.0549 SI, the BIF's 95th percentile):

* gravity χ²/N 0.93; magnetics χ²/N 116 and the joint run χ²/N 116 (64 % of the core at the
  0.0549 cap, the largest anomaly predicted 981 nT of 8,845). A forward check shows why: at
  0.0549 SI even a 2 x 2 x 2 km block under the survey gives only -775 to +670 nT at 80 m
  (a 200 m band needs 1.9 SI for +8,845 nT); the hand samples' induced magnetization
  cannot make these anomalies, so remanence (the MVI cells point at I 58°, D 174°, against
  the field's I 19°, D -1°) or fresh magnetite at depth (two weathered BIF samples) must.
* With κ ≤ 3 SI: magnetics χ²/N 0.95, MVI (±3 SI a component, 112.5 m data, 2,025 x
  306,107 x 3) 0.96, joint 1.01 (gravity 0.97, magnetics 1.01). By volume, 47 % (SI) and
  75 % (MVI) of the cells above 0.055 SI under the stations were deeper than 1 km.
* No sources below 1 km (`max_source_depth_m`, the BIF lying in the top kilometre): χ²/N
  0.94, RMS 129 nT against 126 — the data do not need the deep ones; the strong cells now
  lie at 266 / 644 / 936 m (10 / 50 / 90 % by volume).

Depths of a model's cells are now given by volume: by count the octree's many small cells
near the surface made the strong cells look shallow (91 / 236 / 655 m for the same cells).

MVI and induced runs on the same data shared one Survey node and differed only at the
result. `result_workflow.is_mvi` puts the magnetization in the data key: an MVI run gets a
Survey node (and tree) of its own, named "MVI magnetics · N stations".

## 2026-10-05 — Does a modelled body pull the inversion to its depth? A BIF band at 500 m

The user asked whether a constraint in the starting model makes the inversion put the
anomalies at its depth. For gravity and magnetics (linear) the starting model hardly
matters; the model builder's body acts through the reference model, the smallness weight
(x5) and its bounds (methods/geology.py). A 200 m wide band along the DEM iron-formation
ridge in the window (147° E of N, through 661,256 / 1,667,375), 400-600 m below the ground,
reference 1 SI, sparse 0, 2, 2, 1 with κ in [0, 3], OcTree 8/8/8:

| run | χ²/N | κ in the band at 400-600 m | at 50-250 m | cells > 0.3 SI in the band |
|---|---|---|---|---|
| real data, no constraint (1 km limit) | 0.94 | 0.04 | 0.02 | 0 % |
| real data, soft (0-3 SI, weight 5) | 0.90 | 0.80 | 0.02 | 5 % |
| real data, hard (0.3-3 SI) | 0.93 | 0.81 | 0.02 | 5 % |
| synthetic, true band 1 SI at 50-250 m, no constraint | 0.92 | 0.00 | 0.67 | 0 % (87 % at 50-250 m) |
| the same, soft constraint at 400-600 m | 1.09 | 0.83 | 0.47 | 60 % (26 % at 50-250 m) |

The synthetic data (forward modelled on 25 m cells at the survey's 4,489 stations, 1 % +
10 nT noise) are recovered at the right depth without the constraint; with a wrong one at
500 m the model takes it (60 % of the strong cells in it) and still fits (χ²/N 0.92 →
1.09, RMS 26 → 25 nT): a band 200 m wide at 500 m under a shallow one is beyond what the
data resolve, so whatever the constraint says there is kept. The real data accept the band
just as well (χ²/N 0.90-0.93). A constraint should therefore come from evidence (holes,
mapped units, measured properties) and be tested: with and without, the χ² (here +17 %,
inside the ±10 % discrepancy tolerance only just) and the residual map along the body.

## 2026-10-05 — The deep block under Block 8: the cell weighting, not the norm

After a shallow magnetic layer the runs kept a large weak block below 1 km. None of the
magnetic runs had removed a regional field (the page's default is none); the window's data
hold 31 % of their variance at wavelengths of 5 km and more (66 % above 2.5 km), a plane of
-3,297 to +2,100 nT. Ten runs, κ in [0, 3], 75 m data, OcTree 8/8/8, no depth limit
(deep block: cells > 0.027 SI more than 1 km below the stations' ground; strong: > 0.3 SI,
depths by volume):

| run | χ²/N | deep block km³ | strong at 0-300 m km³ | strong depth 10/50/90 % m | 0-1 km share |
|---|---|---|---|---|---|
| Lp 0,2,2,1, sensitivity, no regional | 0.95 | 4.8 | 0.24 | 365/888/1627 | 3 % |
| the same, plane removed | 0.99 | 8.2 | 0.32 | 309/689/1272 | 4 % |
| smooth L2, plane removed | 1.02 | 22.4 | 1.04 | 327/1015/1989 | 4 % |
| L1 (1,1,1,1) | 1.03 | 16.4 | 0.65 | 314/836/1611 | 3 % |
| MGS | 0.97 | 22.2 | 1.01 | 321/1028/2027 | 3 % |
| TV | 0.83 | 20.9 | 1.06 | 320/1039/2072 | 3 % |
| L1-L2 α 0.8 (IRLS, S1 = ‖k_j‖^½) | 0.93 | 2.0 | 1.90 | 63/173/525 | 43 % |
| Lp, depth weighting β 3 | 1.00 | 15.1 | 0.22 | 473/1072/1822 | 5 % |
| Lp, depth weighting β 2 | 0.82 | 5.7 | 0.40 | 276/651/1177 | 14 % |
| Lp, depth weighting β 1 | 0.72 | 1.9 | 0.91 | 96/325/660 | 36 % |

("0-1 km share": of the magnetization, value x volume, under the stations above 1 km; the
rest lies beside the stations in the padding.) Removing a plane does not take the block
away. With full sensitivity weighting (and Li-Oldenburg β 3) every norm keeps it: these
weights make a deep cell as cheap as a shallow one for its effect on the data, and a broad
weak anomaly is then cheapest as a big deep body. Weaker compensation moves the model up:
β 2, β 1, or the L1-L2's S1 weights (the shallowest bodies, χ²/N 0.93). β 1 and β 2
overfit somewhat (χ²/N 0.72, 0.82). With a 1 km source-depth limit (earlier) the block goes
by construction at the same fit.

## 2026-10-05 — Magnetics: an order-2 trend and depth weighting β 1.5 by default; the display to -3,000 m

Two runs on the Block-8 window settled the defaults (Lp 0, 2, 2, 1, κ in [0, 3], an order-2
trend removed, OcTree 8/8/8, no depth limit; the mesh keeps its deep cells, which only cost
more): depth weighting β 1.5, χ²/N 0.96, deep block 1.37 km³, strong cells (> 0.3 SI) at
171 / 430 / 849 m; β 1, χ²/N 0.95, 4.29 km³, 75 / 292 / 615 m. So a single magnetic
inversion now weighs its cells by depth with β = `MAG_DEPTH_BETA` = 1.5, on the page (the
Weighting field's default for a single magnetic inversion, kept once the user or a restored
setup sets it) and in the worker (when the job does not say, also in the automatic mode);
gravity and joint runs keep the sensitivity weights, and a Bayesian posterior of a magnetic
run without its own weighting starts by depth too. The magnetic data card removes an order-2
trend surface by default (old setups restore their own "none").

The 3D view and the slices stopped at the core (2.5 km below the lowest ground: -1,932 m
here). Both now reach `DISPLAY_BOTTOM_ELEV_M` = -3,000 m (or the core's bottom when deeper,
never below the mesh): `ViewerGrid` samples the padding below a shallow core, and
`ResultModel.line` goes down to the same elevation (by depth below the ground: to the
highest ground on the line + 3,000 m, the cells below -3,000 m left blank).

## 2026-10-05 — A synthetic ablation: no prior, a wrong prior, boreholes only

`examples/output/synthetic_ablation/` (report `ablation_report.html`, in Chinese). A cube (0.5 SI,
100–400 m) and a 45° intrusion (1 SI, 50–1,000 m) under random ground (±150 m, seed 2026), TMI
80 m above it at Block 8's field, noise 2 % + 5 nT, forward modelled on 25 m cells; inverted on
the OcTree 50/25 m, levels 8/8/8 (233,268 cells), κ in [0, 3], depth weighting β 1.5. Six
regularizations × three priors, 18 local runs of 1–2 minutes:

- All fit (χ²/N 0.90–1.38, RMS 12–26 nT); the models differ by 169 m in the cube's depth and 41°
  in the intrusion's dip.
- A, no prior: L2/MGS/TV put the cube's magnetization at 423–435 m (true 250) and the intrusion at
  67–70° (true 45°), κ 4–6 × too low. Lp (0,2,2,1) is closest in shape (266 m, 40°) but packs the
  magnetization into blocks over 2 SI; its correlation with the truth is the lowest (0.21).
- B, the bodies drawn ~100 m off, 50 m deeper, 10° steeper, κ ±50 %: the model builder's default
  bounds (±0.005 SI, weight 5) pin the prior's inner cells (the cube at 0.745, its lower bound;
  the intrusion at 0.495–0.505), so every method returns the prior; the wrong parts keep
  0.37–0.39 SI (A: 0.07–0.17) at the same fit, and the correlation falls for five of six.
- C, five holes (88 cells): held at their logs, but the change from A falls below 0.025 SI
  (mostly below 0.01) beyond 100 m of a hole; a hole 150 m from H5, not given to the inversion,
  sees the same model as in A.

Open: a looser prior (B′: ±50 %, lower weight) and boreholes with a radius of influence (C′); the
model builder's ±0.005 default amounts to a hard constraint and the page does not say so.

## 2026-10-05 — The ablation again: Lp only, tuned, a soft reference model, boreholes with a reach

`examples/output/synthetic_ablation_lp/` (report `ablation_report.html`, in Chinese) replaces the
first ablation, which tuned nothing, used a 100 m sheet as its intrusion, pinned its prior with
the model builder's default range (±0.005 SI) and let its holes reach no further than their trace.
The intrusion is now 250 m thick, 1.2 km long, 100–600 m deep (45° E, 1 SI); Lp only, 40 settings
(norms (0,0,0,0) (0,1,1,1) (0,2,2,1) (0,2,2,2) (1,1,1,1) × depth weighting β 1, 1.5, 2, 3 × length
scale L 1, 3), up to 200 iterations (100 IRLS), in five groups: A none; B the biased prior (~100 m
off, 50 m deeper, 10° steeper, κ ±50 %) as a reference model only (bounds 0–3 SI) at weight 1, 10,
100; C five holes held at their logs and reaching 150 m (the new `radius_m` of borehole sources:
share 1 − d/radius). 200 runs on 10 × c5.4xlarge (`deploy/ec2_sweep.py`, `deploy/sweep_runner.py`):
1.70 h, ≈ $11.6; 199 converged, χ²/N 0.85–1.12. Ranked by the volume-matched overlap with each body.

- Tuning matters more than the prior: in A the score runs 0.23–0.80 (median 0.58). L = 3 beats L = 1
  in every group (A 0.66 vs 0.42 on average); (1,1,1,1) is the most robust norm (A 0.72); β 3 is
  worse. The page's default (p 0,2,2,1, α 1, β 1.5) scores 0.31.
- A's best (1,1,1,1, β 1.5, L 1): overlap 0.83 / 0.77, cube 242 m (true 262 by the same measure),
  dip 55° (45°), κ in the intrusion 0.79 SI.
- The biased reference model at best breaks even and hurts more as its weight grows: best 0.81 /
  0.70 / 0.53 at weight 1 / 10 / 100, better than A at the same setting 25 / 15 / 10 of 40 times;
  at weight 100 the prior's cells sit exactly at their reference, as with the pinned prior.
  Magnetization where the prior is wrong: median 0.38–0.47 SI against A's 0.29, at the same fit.
- The holes help: best 0.87 (dip 47°, κ 0.89 SI), better than A at 35 of 40 settings; their effect
  reaches about 250 m (C − A 0.2 SI at 150–250 m, 0.04 at 250–500 m). A hole 224 m from any other,
  not given to the inversion, sees 0.98 SI in C and 1.16 in A over the intrusion (true 1).

Open: the page's Lp defaults (L 3 or (1,1,1,1)), a smaller radius for background intervals, and
`radius_m` in the page's borehole import.

The report is now in English, for readers new to the test (no comparison with the first ablation),
with a section on how the boreholes do across all 40 settings (`c_scores`, `c_measures`,
`c_sections`, Table 7): they raise the score at 35 settings, by a median of +0.21 where the
inversion alone scores below 0.5 and +0.00 where it scores 0.7 or more; the cube's overlap falls
below 0.4 at 2 settings with the holes against 14 without; the intrusion's dip stays too steep in
most settings (median 64° → 61°). Published privately as an artifact
(https://claude.ai/artifact/SrqZzEDy5cYpgkdJEJQAfW).

## 2026-10-05 — Magnetics: Lp defaults p = (1,1,1,1) at length scale 3

From the Lp ablation (`examples/output/synthetic_ablation_lp`): a single magnetic inversion with Lp
(sparse) regularization now defaults to p = (1,1,1,1) and α_x = α_y = α_z = 3 (length scales),
`worker.MAG_LP_NORMS` / `MAG_LP_LENGTH`, beside its depth weighting β 1.5. On the synthetic model
that setting scores 0.74 without a prior and 0.76 with the boreholes; the former default, p =
(0,2,2,1) at 1, scored 0.31. The worker applies it in the automatic mode and whenever the job leaves
the norms or the length scales out; the page shows it for a single magnetic inversion with Lp (a
new norm preset, "Robust compact"), until the user changes the norms or α, and a restored setup
keeps its own. Gravity, DC/MT and joint inversions, and the other regularizations, keep p =
(0,2,2,1) and α 1: they were not tested. The report says so and was republished (version 2).

## 2026-10-06 — Inputs stored once: clones of an identical file kept before

`~/.geoinv3d` had grown to 14 GB, 12.75 GB of it one file: Block 8's 240 MB TMI grid, copied twice
per job (the server's `inputs/<task>/data` for re-runs and the local backend's
`local/<task>/data`), 54 copies. Those were replaced by APFS clones of one of them (hashes checked
before and after; 12 GiB back on the disk, no job changed), and from now on
`geoinv3d.io.filestore.store` keeps an input as a clone of an identical file stored before
(found by size and hash in `~/.geoinv3d/content_index.json`, checked again before it is shared),
else as a clone of the upload; on file systems without clones it copies. The server's kept
inputs, the local backend's job data and the EC2 staging folder use it; files under 1 MB are
copied as before. Tests keep their index in their own folder (`GEOINV3D_CONTENT_INDEX`).

Also moved to the Trash: ten generated workflow viewers (`*.geoinv3d_viewer.html` and their
`.geoinv3d.json`, 343 MB, not in the repository) of the Karnataka joint, magnetic and gravity-with-
terrain studies and of `examples/output/ec2_runs`; each study's `scripts/build_workflow.py` makes
them again.

## 2026-10-07 — MT: air above the ground, an air-over-background primary, the solver named

MT had no air. The tensor mesh ended at the highest ground and the OcTree was padded below only,
so the secondary field (SimPEG's primary/secondary formulation, zero on the mesh's boundary) was
held to zero at the ground; the cells above the topography took the background conductivity and
the primary was a uniform whole space. Now:

- an MT job's mesh reaches as far above the ground as it is padded around and below
  (`mesh_design.used.air_m`): two core-thick cells, then cells growing by the padding factor
  (tensor), or the OcTree's padding upwards; the cells above the ground are inactive also on a
  tensor mesh without a DEM;
- the inactive cells are air, 1e-8 S/m (`MTMethod(sigma_inactive=...)`, as for DC);
- the primary is a 1D model on the mesh's vertical cells (`MTMethod.primary_1d`): the
  background in the layers whose volume is mostly active, the air above; with every cell
  active, a whole space as before;
- the sparse solver is chosen explicitly (`geoinv3d.methods.solvers.pde_solver`: PARDISO, then
  MUMPS, then SciPy's SuperLU; `GEOINV3D_SOLVER` forces one), recorded as `settings.solver`, with
  a note when it is SuperLU.

Check (`test_two_layers_match_the_1d_solution`, run with `GEOINV3D_SLOW_TESTS=1` or a fast
solver): 10 Ω·m over 1000 Ω·m from 300 m, 22,264 cells. Against the 1D impedance, at 1 Hz the
apparent resistivity is off by -1.1 % with air and +7.1 % with conducting "air"; at 10 and 100 Hz
both are 8 and 16 % off, as 50 m cells are too coarse for a 159 m skin depth (the mesh is next).
SuperLU took 3 minutes for three frequencies on that mesh: neither PARDISO (Intel MKL, not on
Apple silicon) nor MUMPS (no wheels on PyPI, no Homebrew formula) is installed here.

MT and DC on EC2 now solve with PARDISO: the instances install `pydiso==0.3.1` and `mkl==2026.1.0`
(x86 Linux wheels; the Batch image gets pydiso too). Checked on a c5.2xlarge (ap-south-1, 4 min,
about $0.02): the bootstrap took 0.5 min, `pde_solver()` picked Pardiso, and the two-layer case
above (22,264 cells, 71,898 edges) took 3.8 s per frequency against 158.5 s with SuperLU on the
same instance (42x), with the same answers (-1.1 % at 1 Hz). The Mac keeps SuperLU (pydiso has no
wheels for Apple silicon), for small tests.

## 2026-10-07 — MT from EDI files: the impedance tensor, the tipper, ρa and phase

MT data came only as a `.npz` already in the mesh's frame. Field data come as EDI files, one
station each, so:

- `geoinv3d/io/edi.py` reads (and writes) an EDI file's impedance sections: the position
  (LAT / LONG or REFLAT / REFLONG, `dd:mm:ss` or decimal), the frequencies, Z (field units
  mV/km/nT, × 4π·10⁻⁴ to ohms) and its variances, the tipper (`TXR.EXP`...) and its variances;
  data in a turned frame (ZROT, TROT) are turned back to north; an element missing either part
  (the EMPTY value) is missing; spectra-only files are refused;
- `geoinv3d/io/mt_data.py` puts the stations on one frequency list (the distinct frequencies
  within 2 %, or a number per decade), takes the elements asked for in SimPEG's names and
  frame, and gives their errors: the impedance off-diagonal or full tensor, as real and
  imaginary parts or as apparent resistivity and phase, the tipper if any. EDI is x north,
  y east, z down; the mesh x east, y north, z up, so EDI Zxy is SimPEG's Zyx, Zxx and Zyy swap,
  and the tipper changes sign (SimPEG Tzx = −EDI Tzy, Tzy = −EDI Tzx). Each error is the file's
  or a floor, whichever is larger: a share of √|Zxy·Zyx| (5 %) for the impedance, 2ρ times it
  for ρa and its arcsine in degrees for the phase, 0.03 for the tipper;
- `MTMethod` takes a mask (frequency, component, station): a station without a datum at a
  frequency gets no receiver there; the worker reads EDI files of an MT dataset
  (`impedance`, `tipper`, `data_type`, `error_floor`, `tipper_floor`, `frequency_range`,
  `frequencies_per_decade`), projects the stations into the job's UTM zone and puts them on
  the ground;
- the page takes `.edi` files under Electrical → MT, with the choices above shown for MT
  only (the uncertainty row is hidden: the EDI errors and floors replace it); all the EDI
  files of a dataset are read together in the browser for the stations' extent and spacing,
  and the review lists what is inverted and the floors.

Checks (`tests/test_mt_data.py`): a file written and read back, a rotated file turned back,
the frame mapping and the floors; over a 100 Ω·m halfspace the EDI-converted data match
SimPEG's prediction within 1 % for both data types; with a conductor to the east the induction
arrow (Parkinson) points east. In the page, four EDI files went through every step to a job's
parameters, and the worker ran those parameters here: 4 stations × 3 frequencies × 8 components
= 96 data, an OcTree with the air above, three iterations.

## 2026-10-07 — The page: DC resistivity and electromagnetics on cards of their own

DC and MT shared an "Electrical" card with a method switch. DC is an electrical method (a
current put into the ground through electrodes, a static field); MT is an electromagnetic one
(natural fields, induction, frequencies). They only share the property they recover, the
resistivity. Now there are four cards, two rows: gravity and magnetics, then **DC
resistivity** (`.npz` with the electrodes) and **Electromagnetic** (MT for now: `.edi` files or
one `.npz`; controlled-source methods would join it). A dataset's `type` is `dc` or `em`; its
`method` is unchanged (`dc`, `mt`), so the worker is untouched.

- Setups and finished jobs saved with the old card (`type: "electrical"`, `single:electrical`)
  open on the card of their method; an MT dataset's EDI choices are restored too.
- A joint inversion of DC with MT is not offered: both see the resistivity, so they should
  share one model rather than be coupled as two (PGI and the linear correspondence would also
  take them as two properties). DC or MT can still be inverted jointly with gravity or
  magnetics, or with each other as separate jobs.
- EDI files and a `.npz` on the Electromagnetic card together are refused.
- The browser now reads a DC `.npz` (`electrodes`, or `a`, `b`, `m`, `n`) for the extent and
  spacing: a datum at the centre of its electrodes. Before, it said the file could not be read
  and the mesh step had nothing to recommend from.

## 2026-10-07 — MT: the mesh from the skin depths, the fields where they are measured, a layered primary

Step 4 of the MT plan.  Every number below compares SimPEG's impedance with the analytic one of
the model as the cells hold it (each cell under the station a layer), so it measures the
solution's error, not the staircase of the interfaces.

**Where the fields are taken (methods/mt.py).**  E on the top of the ground cell under each
station, H at the centre of the air cell above it (`Impedance(locations_e, locations_h)`;
the tipper's Hz at the station, Hx and Hy above).  H is continuous through the ground but
bends there; SimPEG interpolated it between the cell centres either side, an error of about
half the top cell over the skin depth.  A 1D study of the same scheme (E on nodes, H at cell
centres) and the 3D two-layer case agree: with 50 m cells over 10 ohm m, +7.8 % at 10 Hz and
+16.4 % at 100 Hz before, -0.6 % and -0.1 % after.  The earlier "-1.1 % at 1 Hz" was a
cancellation: its phase was 18 degrees off, the mesh's 5 km of air too thin.

**The mesh (cloud/meshing.recommend_mt_mesh, worker._mt_axes / _build_mt_*_mesh).**  From the
stations' apparent resistivity, 10th/50th/90th percentiles per frequency:
- top cells a quarter of the smallest skin depth (the highest frequency over the 10th
  percentile), three of them, then 10 % thicker a layer to the Bostick depth of the lowest
  frequency: in the 1D study, 1/4 keeps a halfspace within 0.3 % and 1 deg, and growth 1.1
  keeps a 1 ohm m conductor under 1000 ohm m within 4 % (1.2 doubles it);
- padding around and below, and air above, twice the largest skin depth (the lowest
  frequency over the 90th percentile, raised by Niblett-Bostick, (1 + m) / (1 - m), when the
  curve still climbs there: 10 over 1000 ohm m shows 300 ohm m at 0.3 Hz);
- horizontal cells half the station spacing, the core a spacing beyond the outer stations.
An OcTree merges cells in all three directions at once, and the low frequencies sense the ground
far around: merged across the near-surface layering it misrepresents it.  So the OcTree keeps
the base (tensor) cells in every column within the padding from the first air cells as tall as a
core cell down to where the cells are as thick as one (the slab), and over the stations to the
depth of investigation; coarser away from those.  Each axis is a power of two long (cells of
the last size beyond the padding), the ground the middle node of z.  The page asks the server
(`POST /api/mesh/mt`) for the design and its cells, so page and worker share one rule.

**The primary (worker._mt_primaries, meshing.smooth_1d_layers).**  With a uniform primary the
secondary field of a layered earth is the whole layering, laterally uniform: held to zero at the
mesh's edge and carried through coarse OcTree cells it fails at low frequencies.  The primary is
now the smoothest layered earth fitting the stations' median apparent resistivity (Occam's idea:
30 layers, Gauss-Newton on log resistivity, the smoothing lowered until the fit is within 2 %;
0.1 s), `mt_primary` "smooth1d" (default), "bostick" or "uniform"; the result records it.

Checks on EC2 (PARDISO; 10 km array, 5 x 5 stations 2 km apart; worst apparent resistivity error):

| 0.01-100 Hz, OcTree     | uniform primary | Bostick | smooth 1D |
|-------------------------|-----------------|---------|-----------|
| halfspace 100 ohm m     | 0.28 %          | 0.28 %  | 0.28 %    |
| 10 over 1000 (300 m)    | 73 %            | 10.9 %  | 1.3 %     |
| 1000 over 10 (1 km)     | 3.8 %           | 2.4 %   | 2.4 %     |
| 100 / 10 / 1000         | 94 %            | -       | 0.9 %     |

Phases within 0.5 deg with the smooth primary.  A 5 ohm m block in a 100 ohm m halfspace changed
by at most 0.03 % in Z and 0.0001 in the tipper between padding of 1, 2 and 4 skin depths, and
between OcTree coarsenings.  With only 0.3-30 Hz the three-layer earth stays 3.4 % off at 0.3 Hz
(two decades do not constrain its basement, so neither does the primary); likewise from 0.01, 1
and 100 Hz alone 10 over 1000 was 12 % off at 0.01 Hz, against 1.3 % from nine frequencies: the
primary wants the data sampled a few times a decade.  `deploy/mt_accuracy.py` runs these checks on
one instance: the mesh and the primary from nine frequencies, solved at 0.01, 1 and 100 Hz (about
15 minutes; `--full`: all of the above).

Memory: SimPEG keeps every frequency's factorization (`Ainv`) unless `forward_only`; 9
frequencies of a 150,000-cell OcTree did not fit 64 GB.  One at a time: 15 GB and 80 s per
frequency (m5.4xlarge, 150,476 cells), 23 GB and 160 s (221,184-cell tensor).  For inversions
this is step 5 (frequencies one at a time, or in parallel).

## 2026-10-07 — A 3D model builder of its own page

Modelling now has its own page, `/model` (`geoinv3d/viz/model_builder.html`, three.js 0.160 from
jsDelivr): a geological model in 3D, edited with the mouse, saved on the server, and opened by
the upload page's model step as an inversion's constraints.  It keeps the model step's format
(`builder` version 2: bodies — box, cylinder or polygon between two depths or elevations,
straight down or dipping — and layer stacks, everywhere or inside an outline, their interfaces
tilted; each part with a density, susceptibility and/or resistivity, its range, weight, fixed and
sharp), so nothing changed downstream: the upload page turns it into the spec of
`geoinv3d/methods/geology.py` (reference model, bounds, smallness weights by volume share) as
before.  This is MARE2DEM's separation of the model (its .poly regions) from the meshes it is
put on, in 3D.

The page: the area (extent, ground, depth shown, vertical exaggeration, CRS); bodies and stacks
added, drawn (a polygon clicked on the ground in plan view), moved and lifted with a gizmo,
boxes and cylinders resized, polygon vertices dragged, inserted, deleted; dip and dip direction;
a values table per body and per layer; the background ("everywhere else"); colour by unit or
by a property (resistivity on a log scale) with a colour bar; plan, south and east views; a
west–east or south–north section (clipping); undo / redo; save, save as, open, delete; JSON out
and in (the model step's own ⤓ Save model files too).  "Use in an inversion" saves and opens
the upload page with `?model=<id>`, whose model step takes the model; the step also lists the
saved models ("From the model builder…").

Server: `GET /model`; `GET/POST /api/models`, `GET/PUT/DELETE /api/models/{id}`, the models in
`~/.geoinv3d/models` (`GEOINV3D_MODELS_DIR`); a deleted model goes to its `deleted` folder.

Next: forward data from a model (gravity, magnetics, DC, MT on the job's mesh), so that a model
built here also makes synthetic surveys.

## 2026-10-07 — Boreholes logged by interval, in the model builder and the constraints

A borehole is now a list of depth intervals, each with its own density, susceptibility and/or
resistivity (and a lithology), not one unit for the whole hole.

- `geoinv3d/methods/geology.py`: a "boreholes" source may give its holes inline
  (`holes: [{name, x, y, collar_m, azimuth, inclination, intervals: [{from_m, to_m, unit}]}]`,
  depths along the hole from the collar, the collar on the ground unless given).  The cells the
  hole passes through take the intervals' units by their length in them (points every quarter
  of half the thinnest cell: 51.5 ohm m where the exact log mix of 30 m at 60 and 20 m at 40 ohm m
  is 51.0); with `radius_m`, the cells around take the longest interval of the nearest hole
  cell, in part, as before.  The file-based holes (one unit per hole) are unchanged.
- The model builder: "⊙ Boreholes" (a set: reach, weight, fixed, sharp; a holes table — name,
  E, N, collar, azimuth, dip — and the picked hole's intervals: from, to, lithology, the three
  values); "⤒ Boreholes CSV" and a pasted CSV, one row per interval (`hole,x,y,from,to,…`) or per
  measurement (`hole,x,y,depth,…`: each value then stands for the depths half-way to its
  neighbours), headers with units in brackets accepted, `*_lower` / `*_upper` columns for ranges;
  the area widens to take in the holes.  Holes are drawn along their azimuth and dip, each interval
  coloured by lithology or by the property shown, the reach as a faint sleeve; "◎ Zoom to selected"
  (or a double-click) brings a body or a hole into view.
- The upload page keeps a model's borehole sets apart from its editor (`MB.holes`), shows the
  collars on its map, and makes every interval with a value a unit ("Boreholes 1: BH1 0–30 m") of a
  "boreholes" source of each property's spec; its "⤓ Save model" writes them back.

Checked: tests of a vertical hole crossing an interval boundary inside a cell, of an inclined hole
with a collar above the ground and of a reversed interval; in the browser, a CSV of four depth
measurements becomes four intervals with midpoint boundaries, the upload page's spec of the saved
model carries a borehole source per property, and that spec, placed by geology.py on 50 m cells,
puts the logged resistivities down the right columns (the inclined hole moving east with depth).

## 2026-10-07 — The upload page's Model step is the 3D builder

The Model step's own editor (a plan-view map with a section, a list, forms and a layer table,
some 800 lines) is gone; the step now holds the 3D model builder in a frame
(`/model?embed=1`). The page keeps the model (`MB.items`, `MB.holes`, `MB.free`, `MB.ground`)
and makes the jobs' specs from it as before (`mbSpec`, `geoCombined`, `mbImport` and "⤓ Save
model" unchanged); the frame and the page talk by `postMessage` (same origin only):

- page → builder `geoinv3d:context`: the area (the data window), the ground (flat, or the
  model's), the CRS, the mesh's core depth (shown depth, first time) and the data — each type
  as at most ~20,000 points (grids sampled to 150 × 150) with its colour range — which the
  builder draws on its ground (Data, Viridis, with a legend), so bodies are drawn over the
  anomalies as on the old map; the area fields are then the page's, not editable there;
- page → builder `geoinv3d:load`: a model from elsewhere (`?model=<id>`, a setup, a finished
  job, a spec dropped on the files area — which now takes a file with only a `builder`);
- builder → page `geoinv3d:ready`, and `geoinv3d:model` after every change (not after a load
  from the page, and not for its own empty start, so it never wipes the page's model).

Left out of the new editor for now: pasting a layer table, the section with the mesh's cells
and the thin-layer flags, and moving a layer within its stack.

Checked in the browser: a 30 × 30 synthetic gravity anomaly on the builder's ground, the area
following the window (400–410 km E), a box and a borehole added there reaching the page's
"⤓ Save model" spec and the submitted `params.geology` (a body and a boreholes source), a saved
model through `?model=` and a dropped builder JSON appearing in the frame, every step without a
script error.

## 2026-10-07 — The builder's missing pieces; a builder model's data inverted for all four methods

**The model builder** got back what the old Model step had (and the 3D one lacked):

- *Paste a table of layers* under a stack's table: `[name] thickness value [lowest highest]`
  per line (a name may be several words; a first line of words only is a header), for the
  property chosen there, `-`/`inf` as the last thickness reaching the floor; "Replace the
  layers" or "Add below". What it set by itself (a middle `-` → 100 m, the old floor layer
  given a thickness, a last layer with one) is said in the message; skipped lines too.
- ↑ ↓ on each layer row, and the list's ↑ ↓ with a layer selected, move it within its stack;
  the layer reaching the floor stays last (the two swap thicknesses when it moves).
- The mesh's cells on sections: in the upload page's Model step the job's (`mbMesh`: the
  tensor core, an OcTree's finest levels, an MT mesh's top cells growing by `z_growth`; sent
  with the context), on the builder's own page uniform cells typed under the area (saved with
  the model, `mesh: {dx, dz}`, kept by `_clean_model`). A layer or body thinner than the cells
  at its depth, or a body narrower (its least width over all directions, so a diagonal dyke
  too), gets a ⚠ with the reason in the list (each layer's row) and its form.

**`geoinv3d/methods/builder.py`**: the page's `mbSpec` / `mbUnit` in Python (`builder_spec`;
`mbUnit`'s weight now falls back to 5 for a null weight on both sides) and a property's values
on cells (`model_values`, the spec's reference model: volume averages) — a true model from a
drawn one.

**`examples/synthetic_builder.py`**: a 2 km × 2 km builder model (a 50 m 20 Ω·m cover over
1000 Ω·m, a sulphide body dipping 60° E — 3.3 g/cc, 0.05 SI, 3 Ω·m, 100–400 m — and a granite
stock, 2.57 g/cc, 5000 Ω·m, 150–800 m) and two holes logged through them. Its data (gravity and
TMI on a 100 m grid from a 50 × 50 × 25 m mesh; DC dipole-dipole a = 100 m, n = 1–4, five
lines; MT xy/yx at 4 × 4 stations, 10/100/1000 Hz; DC and MT on the inversion meshes — an
inverse crime) go through `run_data_pipeline` on coarse meshes (100 × 50 m cells to 800 m; MT
200 m wide, 25 m at the top growing 10 %), free and with the holes' logs as constraints
(reach 150 m). Correlation with the true model over the core (to 600 m; resistivity in log):

| | χ²/N free / holes | corr. free / holes | sulphide (true) free / holes | time |
|---|---|---|---|---|
| gravity (sparse) | 1.10 / 1.05 | 0.74 / 0.72 | 0.32 / 0.32 (0.55 g/cc) | 2 s |
| magnetics (sparse) | 1.11 / 1.14 | 0.67 / 0.72 | 0.014 / 0.018 (0.044 SI) | 2 s |
| DC (L2) | 1.66 / 0.98 | 0.26 / 0.29 | 2.07 / 2.03 (0.79 log Ω·m) | 52 s |
| DC a = 50 m, n = 1–8 (25 m cells, EC2) | 1.06 / — | 0.08 / — | 1.81 / — | 16 min |

Gravity puts the body in the right place (its density spread over the coarse cells); the holes
pull the magnetic body up where the free inversion smears it down.

**Why DC correlates so poorly** (the a = 100 m run; the depth of investigation from the same
data inverted from 20 and from 1000 Ω·m, Oldenburg & Li 1999): the data decide the model to
~200 m (DOI index 0.1–0.2 at 50–200 m, 0.44 at 200–300 m, 0.9–1.0 below), 29 % of the core;
below 300 m — half the core — the model is its reference, 63 Ω·m, where the truth is 1000 Ω·m.
Even where the data decide, the correlation is 0.30: under the 20 Ω·m cover the current stays in
it, and the resistive basement's value hardly matters to the data (50–150 m: 2.95 true, 1.87
recovered, log Ω·m); and the cover is one 50 m cell, smoothed into the basement (47 Ω·m for
20). The recovered model's spread is 0.22 decades for the truth's 0.59. Shorter dipoles
(a = 50 m, n = 1–8, on 25 m cells) resolve the cover but see less deep: 0.08. What would help:
long offsets (a 200 m set beside the 50 m one; `dc_multi` in the example, not yet run), a
layered background from a 1D inversion of the soundings (as MT's smooth1d primary) instead of a
constant, thinner top cells growing with depth, a joint inversion with MT (deep) — and judging
the model where the DOI is small. (From 1000 Ω·m, 8 iterations left χ²/N at 1539: a start far
from the data is slow to recover — the background from the data is the right default.)

MT: the plumbing ran locally (synthetic data: 348 s with SciPy's SuperLU; the same on EC2 with
PARDISO, m5.4xlarge: 12.7 s), but each new model costs ~6 minutes of factorizations here, and
the two-iteration runs had not finished after 50 minutes; the EC2 run (the DC variants first)
was stopped to make MT's frequencies parallel first (next). `--ec2` runs the example on one
instance and fetches its outputs (`--attach` follows a run started before; a bare `tail` with
no file waited on the SSH channel's input and hung the first follower).

**The DC / MT background from the data** (`worker._backgrounds_from_data`). The page never set
`sigma_background`, so DC and MT jobs started from, and were pulled towards, 100 Ω·m. Over
this model's 20 Ω·m cover the DC start was 27 000 times its noise and two iterations left it at
3878; from the median of the data's apparent resistivity (63 Ω·m: K·V with the halfspace
geometric factor, poles by NaN electrodes; MT: √(ρxy ρyx)) at 59. Datasets sharing a model label
share one background from all their data (a group lasso refuses a shared model whose datasets
differ); a job's own `sigma_background` stays (≤ 0 is an error); without apparent resistivities
(MT tipper only) 100 Ω·m. Recorded per dataset in the result (`datasets[i].background`).

Reviewed (a reviewer and a web-test agent): the builder spec matches the page's item by item;
the page's mesh context is keyed on all its rows (an OcTree's levels changed nothing in its
label). Tests: `tests/test_builder.py` (new), the background and the models' mesh field in
`test_data_pipeline.py` / `test_api_local.py`; the suite 615 passed, 2 skipped.

## 2026-10-08 — MT's frequencies on processes side by side (step 5, in progress)

Each MT frequency is a PDE of its own, factorized once per model and independent of the others.
SciPy's LU is single-threaded: here a model costs ~2 minutes per frequency on 28k cells, and the
synthetic example's MT runs did not finish (2026-10-07). Solving the frequencies on processes
at once is the first speed-up, and the same split is what an MPI / cluster version would
distribute (one process per frequency group; only vectors travel).

Done (committed, tested):

- `geoinv3d/methods/parallel.py`: `ParallelMetaSimulation` — simulations (one per group of
  frequencies) on processes started with "spawn" (forking a process whose MKL / OpenMP threads
  run can hang), as daemons, each with its share of the cores' threads; the model goes to them
  once, then the fields, predicted data, J v, J^T v and diag(J^T J) are computed by each for its
  frequencies and added up here; a process's error is raised here (`WorkerError`, with its
  traceback); `close()` (also when collected) stops them. The simulations are pickled in the
  main process, so a pickling error is raised at once (a queue's feeder thread only prints it,
  and the process then waits for ever: the first test hung that way).
- `MTMethod(n_workers=1 | k | "auto")`: k contiguous groups of the frequencies with data (the
  data stay frequency by frequency), each a `Simulation3DPrimarySecondary` mapping the model
  itself; "auto" = one per frequency up to the cores on meshes of ≥ 20 000 cells (below,
  starting processes costs more than it saves). Default 1: unchanged behaviour.
- `solvers.pde_solver`: pymatsolver builds `SolverLU` at run time with module "abc", so no
  simulation holding it could be pickled; it is named `pymatsolver.SolverLU` now.
- `tests/test_mt_parallel.py`: two processes give one process's data, J v, J^T v and
  diag(J^T J) to 1e-8 on a masked 3-frequency survey, and fresh fields for a new model; an error
  in a process reaches the caller and the processes keep working; `n_workers` checks.

Still to do:

1. The pipeline: a job's `mt_workers` (default "auto") into each MT dataset's
   `method_kwargs["n_workers"]` (worker.py, where `_mt_primaries` fills the MT kwargs); record the
   processes used in `result["settings"]`; close the simulation when the job ends (joint
   inversions too: `make_simulation_mapped` passes a mapping, untested in parallel).
2. Measure: `examples/synthetic_builder.py OUT --only mt` with the frequencies in parallel
   (3 frequencies: expected ~3x with SuperLU), then the full MT run; on EC2 with PARDISO
   check that the threads per process (cores / processes) do not oversubscribe.
3. Memory: each process keeps its own factorizations (the same total as one process); with
   many frequencies on a big mesh, cap the processes by memory as well as cores.

## 2026-10-08 — MUMPS on the Mac: MT and DC 50 times faster here

pydiso (PARDISO) is Intel MKL, x86 only: on this Mac (Apple M5 Pro, arm64) MT and DC used SciPy's
single-threaded LU. pymatsolver 0.4 also takes MUMPS (`python-mumps`), and `pde_solver()` already
prefers it to SciPy; PyPI has no arm64 wheels of python-mumps, conda-forge has. So a
conda-forge environment (micromamba from Homebrew: `~/mamba/envs/geoinv3d-mumps`, Python 3.13,
SimPEG 0.25.2, discretize 0.12.0, python-mumps) with the rest of the project installed by uv
(`-e ".[full,dev,cloud]" rasterio`); the project's `.venv` is left as it was.

The synthetic example's MT mesh (28 392 cells, 91 000 edges), one forward with the layered
primary:

| solver | 3 frequencies (10, 100, 1000 Hz) | 1000 Hz |
|---|---|---|
| SciPy LU (`.venv`) | 348 s | 110.2 s |
| MUMPS (this Mac) | 7.5 s | 2.2 s |
| PARDISO (EC2 m5.4xlarge) | 12.7 s | — |

MUMPS and SciPy's LU agree to 7.7e-16 (relative). `GeoInv3D.command` now starts the server with
`GEOINV3D_PYTHON` if set, else this environment if it exists, else `.venv`, and says which
solver it has; local jobs run with the server's Python, so they get MUMPS too. The environment
has newer NumPy / SciPy than the EC2 workers' pins (2.5.3 / 1.18.1 against 2.3.3 / 1.16.2).
The whole suite in this environment: 620 passed, none skipped (5 min 36 s).

## 2026-10-08 — MT's processes from the job (step 5, item 1); SuperLU's complex LU on Windows

Item 1 of the list above, on the Windows laptop (Core Ultra 7 265U, 14 threads):

- `mt_workers` (job key, default "auto") becomes each MT dataset's `method_kwargs["n_workers"]`
  unless the dataset gives its own (`worker._mt_workers`, next to `_mt_primaries`); "auto" or a
  whole number from 1, else a ValueError.
- The inversion runs inside `parallel.closing_opened()`: every `ParallelMetaSimulation` started
  in that context (a contextvar: a job running in a thread closes only its own) is closed when
  the inversion ends or fails; SimPEG's objects refer to each other, so collection alone could
  leave the processes waiting for the garbage collector.
- The result records `settings.mt_processes` (1: one process) and, above 1,
  `settings.mt_threads_per_process`, from the simulations actually started.
- Joint inversions: each method gets its slice of the joint vector through a `Wires`
  projection; on two processes the data, J v, J^T v and diag(J^T J) match one process to 1e-8,
  and the density part of J^T v stays zero.

Tests: `test_mt_parallel.py` (+2: the joint slice; the processes stopped after a failing job,
and counted), `test_data_pipeline.py::TestEMData` (+2: `mt_workers` into the datasets and its
checks; a single MT job and a joint gravity + MT job with the cross-gradient on two processes,
`mt_processes` recorded, no process left; skipped with SuperLU unless GEOINV3D_SLOW_TESTS).

On this machine SciPy's SuperLU is unusable for MT: the parallel test's 6,144-cell system (20,536
complex unknowns) did not factorize in 6 minutes. A 3D Laplacian of 19,683 unknowns takes 8.4 s
real and 201 s complex (fill 124x, COLAMD); 2,744 unknowns 0.11 s vs 2.2 s: complex is 20-24x
slower than real, not the 4x of the arithmetic. With `pydiso==0.3.1` and `mkl==2026.1.0` (the
EC2 versions; win_amd64 wheels on PyPI, ~210 MB with intel-openmp) `pde_solver()` picks
Pardiso and the parallel test runs in about 2 minutes, while a magnetic study ran beside it.

Item 2, measured here (PARDISO, 14 threads): `examples/synthetic_builder.py OUT --only mt --quick
--mt-workers 1 | 3` (new option; the result records `mt_processes`). The MT mesh: 28,392 cells
(19,604 below the ground), 3 frequencies (10, 100, 1000 Hz), 192 data, 2 Gauss-Newton iterations:

| run | 1 process | 3 processes (4 threads each) | speed-up |
|---|---|---|---|
| free | 733 s | 327 s | 2.2x |
| with the holes | 748 s | 276 s | 2.7x |
| the whole script (data too) | 1,567 s | 663 s | 2.4x |

The same misfits iteration by iteration (phi_d 2.17e5, 2.64e4, 6.19e3) and the same results (chi2/N
32.24 / 34.45, correlations 0.515 / 0.611, the bodies' values) on both. Less than 3x because PARDISO
is threaded: one process already used the 14 threads. Memory, sampled every 2 s: one process
peaked at 9.15 GB, three at 8.41 GB in all (the largest 2.71 GB): each process keeps only its own
frequencies' factorizations, about 3 GB a frequency on this mesh, so the processes do not add
memory here.

Still to do: item 2 on EC2 (threads per process with PARDISO on more cores); item 3 (memory):
since the total does not grow with the processes, a cap by memory matters only beside a mode that
keeps fewer factorizations (frequencies one at a time) for meshes whose frequencies do not fit at
once (9 frequencies of 150,000 cells did not fit 64 GB).

## 2026-10-08 — MT's frequency groups on MPI ranks, for a cluster

The user asked for an MPI interface, to run on a supercomputing centre later. The split is the
one of the processes (a group of frequencies per worker; the model sent once, then only
vectors), so `methods/parallel.py` now has one worker loop (`_Server`: simulations held by id,
their model and fields; the messages "sim", "model", "fields", "dpred", "jvec", "jtvec", "jtj",
"drop") and two transports for it:

- `processes` (as before): spawned daemon processes, one per group, each with its share of the
  cores' threads;
- `mpi` (mpi4py): the ranks of the job. Started on N + 1 ranks (`mpiexec -n 4 python -m
  geoinv3d.cloud.worker --local params.json DATA OUT`, or `srun`), `worker.main` runs under
  `parallel.mpi_run`: rank 0 runs the job as usual, ranks 1..N wait in `serve_mpi` and each
  takes a group of every parallel simulation the job makes (several at once share the ranks, by
  id; only rank 0 talks to them, one operation at a time). At the end rank 0 lets them go; an
  error on rank 0 calls `Abort`, so the ranks do not wait until the scheduler's wall time.
  Threads per rank come from the launcher's OMP_NUM_THREADS / MKL_NUM_THREADS.

`parallel_backend()`: GEOINV3D_PARALLEL "processes", "mpi" or "auto" (default: "mpi" when an MPI
launcher started the process on more than one rank: OMPI_COMM_WORLD_SIZE, PMI_SIZE or
SLURM_STEP_NUM_TASKS above 1; nothing imports mpi4py otherwise). Under MPI, `MTMethod.workers_for`
gives "auto" the serving ranks whatever the mesh (they were started for it), at most one per
frequency, and a number no more than there are. The result records `settings.mt_parallel`.
Job scripts: `deploy/hpc/slurm_mt_mpi.sh`, `deploy/hpc/pbs_mt_mpi.sh` (PBS Pro, as Imperial's RCS).

Tests (`tests/test_mt_mpi.py`): the ranks as threads behind a stand-in communicator that pickles
every message as MPI does — the data, J v, J^T v and diag(J^T J) of 2 ranks equal one process's
to 1e-8; two simulations alive on the same ranks; a rank's error raised on rank 0 with its
traceback, the ranks still serving; a closed simulation dropped on them; `mpi_run`'s exit code,
the ranks stopped, and the abort. `test_under_mpiexec` runs `tests/mpi_mt_check.py` on 3 real
ranks when mpi4py and mpiexec are there. The processes' tests, the pipeline's MT tests and the
local backend pass unchanged (22).

With `mpi4py==4.1.2` (the user agreed; a 1.7 MB wheel on Microsoft MPI 10.1, already installed
here, which sets PMI_SIZE): `test_under_mpiexec` passes on 3 ranks (40 s), and a whole job ran as a
cluster would run it — `mpiexec -n 4 python -m geoinv3d.cloud.worker --local params.json data out`,
the synthetic builder's MT data on its 200 m mesh (28,392 cells), 2 iterations: rank 0 ran the
pipeline, "3 frequencies on 3 MPI ranks (4 solver threads each)", exit 0 with every rank gone,
`settings.mt_parallel = "mpi"`, `mt_processes = 3`; 5 min 18 s, as the 3 local processes on that
mesh (327 s for the same 2 iterations).

Also on EC2 (m5.8xlarge, 32 vCPU = 16 cores with hyperthreads; MT inversion mesh 100 x 12.5 m,
88,200 cells; 3 frequencies): one model's fields took 61.9 s on one process (32 threads) and 54.5 s
on three (10 threads each), 1.14x — PARDISO already scales over the cores, unlike the laptop's
2.2-2.7x. The three processes' 30 threads exceed the 16 physical cores; threads per process by
physical cores is still to try.

## 2026-10-08 — The synthetic builder's MT on a finer mesh, on EC2

`examples/synthetic_builder.py OUT --only mt --mt-cell 100 12.5 --mt-data-cell 50 12.5 --mt-timing
--ec2 --type m5.8xlarge` (new options: the inversion mesh, the mesh the data are modelled on —
before, the inversion's own, an "inverse crime" — and the timing of one model's fields).
ap-south-1, Owner=miaozhou, 11:21-12:28 (67 min, about $2), terminated after the fetch; outputs in
`examples/output/mt_parallel_timing/ec2_m5.8xlarge_100m/`.

- Data on 50 x 12.5 m (217,800 cells), 3 frequencies on 3 processes: 328 s.
- Inversion mesh 100 x 12.5 m (88,200 cells, 61,740 below the ground); one model's fields: 61.9 s on
  one process (32 threads), 54.5 s on three (10 threads each).
- Free: 10 iterations, 1,884 s, chi2/N 0.46 (the beta schedule halved past the target),
  correlation with the truth 0.14; the sulphide body 125 ohm m (true 6), the granite 91 ohm m
  (true 4,600). With the two holes: 9 iterations, 1,558 s, chi2/N 1.00, correlation 0.21, 52 and
  209 ohm m. On the 200 m mesh with the data from that mesh the correlations had been 0.52 / 0.61
  (2 iterations): the inverse crime flattered them. 16 stations 500 m apart with 3 frequencies do
  not resolve bodies 250-500 m wide; the section is near-surface speckle where the free run
  over-fits.

The follower stopped with the session that started it (the instance went on); `--attach` picked
the run up, fetched it and terminated the instance. Its iteration lines were hidden below
iteration 10 (the line was stripped before the check); fixed.
