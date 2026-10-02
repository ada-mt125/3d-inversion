# GeoInv3D

**DAG-based 3D Geophysical Joint Inversion Framework**

A Python framework for 3D geophysical joint inversion that uses a Directed Acyclic Graph (DAG) to track every parameter change, making inversion workflows fully reproducible and inspectable.

## Key Features

- **DAG-based workflow tracking** — Every step (mesh creation, model setup, forward modeling, inversion, parameter tuning) is a node in a DAG. Every parameter change is recorded automatically.
- **SimPEG backend** — Uses [SimPEG](https://simpeg.xyz/) for forward modeling and inversion, supporting gravity, magnetics, DC resistivity, and joint inversion.
- **Joint inversion** — The coupling is a choice of its own, separate from each model's regularization: cross-gradient, joint total variation, linear correspondence, petrophysically guided (PGI, rock units), group lasso (Utsugi 2025) or none, with a unit-free coupling weight; each model regularized as smooth L2, sparse lp, L1–L2, MGS or TV with bounds; several datasets may share one model.  See [docs/joint_couplings.md](docs/joint_couplings.md) and [docs/group_lasso_joint.md](docs/group_lasso_joint.md).
- **Workflow serialization** — Save/load entire inversion workflows as JSON. Every exported result carries a provenance sidecar recording exactly how it was produced.
- **Visualization** — 2D slices (matplotlib), 3D models (PyVista/VTK), DAG graph rendering (networkx), and Mermaid diagram export.
- **Immutable data model** — All data payloads are frozen dataclasses with read-only arrays. Transformations produce new objects, never mutate.

## Architecture Overview

```
geoinv3d/
├── core/        DAG engine: Node, Graph, registry, serialization
├── datamodel/   Immutable data: Mesh3D, PhysicalModel, SurveyData, InversionResult
├── methods/     SimPEG wrappers: gravity, magnetics, DC resistivity, joint
├── nodes/       DAG node types: input, transform, forward, inversion, export
├── io/          File I/O (NumPy, JSON)
└── viz/         Visualization: DAG plots, model slices, convergence, 3D
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design.

## Installation

```bash
pip install -e .                    # Core (SimPEG + NumPy)
pip install -e ".[viz]"             # + PyVista for 3D visualization
pip install -e ".[full]"            # + networkx for DAG plots
pip install -e ".[dev]"             # + pytest, ruff, mypy
pip install rasterio                # GeoTIFF grids in the raw-data pipeline (optional)
python -m pytest                    # full test suite, ~3 min
```

### Dependencies

| Package    | Purpose                        |
|------------|--------------------------------|
| numpy      | Arrays, numerics               |
| scipy      | Optimization, sparse matrices  |
| SimPEG     | Forward modeling & inversion   |
| discretize | Mesh generation                |
| matplotlib | 2D plotting                    |
| numba      | Fast coordinate descent (L1–L2)|
| rasterio   | GeoTIFF reading (optional)     |
| pyvista    | 3D visualization (optional)    |
| networkx   | DAG graph rendering (optional) |

## Quick Start

```python
import numpy as np
from geoinv3d.core import Graph
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelCreateNode, SurveyCreateNode
from geoinv3d.nodes.forward_nodes import ForwardNode
from geoinv3d.viz import plot_dag

# Build a workflow DAG
graph = Graph()

# Create mesh
mesh = MeshCreateNode(nx=20, ny=20, nz=10, dx=100, dy=100, dz=50)
graph.add(mesh)

# Create density model
model = ModelCreateNode(mesh, value=1.0, prop="density", name="Density")
graph.add(model)

# Create survey with random station locations
locs = np.column_stack([
    np.random.uniform(0, 2000, 50),
    np.random.uniform(0, 2000, 50),
    np.zeros(50),
])
survey = SurveyCreateNode(locs, np.zeros(50), np.ones(50) * 0.01, method="gravity")
graph.add(survey)

# Forward modeling
fwd = ForwardNode(model, survey, method_type="gravity", name="Gravity Forward")
graph.add(fwd)

# Visualize the DAG
plot_dag(graph, title="My Gravity Workflow")

# Change parameters — the DAG tracks it
model.set_value(2.5)  # invalidates downstream nodes
fwd.evaluate()        # recomputes only what changed

# Save the workflow
from geoinv3d.core import save_workflow
save_workflow(graph, "my_workflow.geoinv3d.json")
```

## Regularization and Trade-off Choices

Single-method inversions (`InversionTask`, the upload page's *Manual* mode, and
the `RegularizedInversionNode` DAG node) offer:

| `regularization_type` | What it does | Key options |
|---|---|---|
| `l1l2` | L1–L2 elastic net (Utsugi 2019). Default solver: coordinate descent along a λ path, λ from the L-curve | `l1_ratio` (α), `l1l2_solver` (`cda`/`irls`), `l1l2_weighting` (`S1`/`S2`), `lambda_decades` |
| `mgs` | Minimum gradient support focusing (sharp, blocky bodies) | `focusing_percentile`, `focusing_scale`; an upper bound is advisable |
| `tv` | Total variation of the model gradient | same as MGS |
| `sparse` | lp-norm IRLS (SimPEG `Sparse`) | `norms`, `alpha_*` |
| `l2` | Smooth L2 | `alpha_*` |

`beta_selection` picks the trade-off parameter: `auto` (L-curve on the L1–L2 λ
path, falling back to χ² = N when the curve has no corner; χ² = N otherwise),
`discrepancy`, `lcurve` or `gcv`. The viewer's *Convergence* tab draws the
L-curve, χ²/N and GCV curves with each criterion's pick.
See `examples/l1l2_paper_synthetic_report.md` for a comparison on a magnetic synthetic.

### How joint models are coupled

`coupling` in a joint job (the upload page: Inversion step, *How the models are
coupled*): `cross_gradient` (structural), `joint_total_variation` (structural, convex),
`linear_correspondence` (a linear rock-physics relation), `pgi` (rock units as a
Gaussian mixture; the result also classifies the cells), `group_lasso` (joint sparsity,
below) or `none`. `coupling_weight` is unit-free (1 = as strong as the regularization):
the former raw cross-gradient weight of 1 coupled nothing. See `docs/joint_couplings.md`.

### Joint gravity–magnetic inversion with a group lasso (Utsugi 2025)

`coupling="group_lasso"` (formerly `regularization_type="group_lasso"`, still accepted)
on a joint job with one gravity and one magnetic dataset minimizes ½‖b − Zζ‖² + λ1 Σₖ √(βₖ² + ρₖ²) + ½λ2‖ζ‖² by ADMM, so
that the density and magnetic models share their support without either being
forced non-zero. The solver, `geoinv3d/methods/group_lasso.py`, takes any
sensitivity operators (`joint_group_lasso_admm`, `JointGroupLassoProblem`, or
`from_simulations` for two SimPEG simulations). What follows the paper and what
is engineering, the parameters and the validation: `docs/group_lasso_joint.md`.

### Geology constraints, topography and coordinates

- **Geology constraints** (`params["geology"]`, or the upload page's *Starting &
  reference model*): rock units from sample densities or given values, placed by samples,
  boreholes, interpreted bodies or map polygons, become the reference model, per-cell
  bounds and smallness weights of a single gravity, magnetic, DC or MT inversion. On the
  page, bodies and layer stacks can be drawn on the data map or typed in (outline, depths
  or elevations, dip or tilted interfaces, a density, susceptibility and/or resistivity per
  part; layer tables can be pasted), checked on a section coloured by value, and
  saved/loaded as JSON. Cells hold the volume average of what they contain, so layers
  thinner than the cells are not lost; "sharp" parts relax the smoothing across their
  boundaries (like ModEM's covariance tears). See `docs/geology_constraints.md`.
- **Topography**: a DEM (any CRS), x/y/z points, or the data's own station elevations
  (`topography: {"from_data": true}`); the mesh follows the ground and the viewer's 3D
  view draws the terrain and the stations.
- **Coordinates**: tables in longitude/latitude are projected to the job's CRS (the
  GeoTIFF's, else the UTM zone of the data, or `crs`).
- **Bouguer check**: for a gravity station table with elevation, observed and normal
  gravity, the page and the worker report the reduction density and whether a terrain
  correction was applied (`geoinv3d/methods/bouguer.py`).

## Examples and the Viewer

```bash
python examples/regularization_viewer_demo.py     # ~5 min: all choices as DAG nodes
python -m geoinv3d.viz.serve_dag examples/regularization_comparison.geoinv3d.json --serve
python examples/l1l2_paper_synthetic.py           # Utsugi-protocol alpha scan, ~45 min on 4 cores
python examples/focusing_comparison.py            # MGS / TV / sparse / L2 vs L1–L2, ~2 min
python examples/lambda_selection_benchmark.py     # discrepancy vs L-curve vs GCV, ~30 min
```

The generated `*_viewer.html` files open directly in a browser (no server needed).
Outputs go to `examples/output/<example>/` (tables, JSON and figures).

## The Upload Page: Inversions from the Browser

```bash
python -m geoinv3d.api            # then open http://127.0.0.1:8000/
```

Drop the data on the page (or start from a saved setup, or from a finished job with its
files), check the area, mesh and model, choose the inversion, and run it **on this computer** (free, no set-up; jobs queue one at a time, `--local-jobs N` for
more) or **on AWS** (EC2 or Batch, with this machine's AWS credentials). Before
submitting, the page estimates the run time, the memory and (on AWS) the cost; the rates
come from the jobs that finished here, so the estimates improve with use. *Preview first*
runs the job with cells twice as large on this computer, and its card then starts the
full-resolution run, here or on AWS, with one click.

While a job runs, its card draws the data misfit against its target (chi^2 = N) and beta
at every iteration; at the end it says whether the run converged or why not (iteration
limit, IRLS limit) and offers the fix. Any finished job can be run again on its own files
with changed settings (iterations, norms, weighting, bounds, mesh, data errors, or all
parameters as JSON), from its card or from its node in the workflow; the new run joins the
workspace's workflow next to the old one. Map layers (GeoJSON or CSV: mines, towns,
geological outlines) are drawn over the 3D view, the depth slice, the sections and the
data-fit maps; a workspace keeps them for all its runs, and a workflow file can carry its
own (`map_layers`).

Before inverting, the Area step shows **the data as interpreters look at them**
(`POST /api/enhance`, `geoinv3d/methods/enhance.py`, after e.g. Yang et al. 2026, Ore Geol.
Rev. 198, 107581): the field reduced to the pole (damped near the magnetic equator),
continued upwards, its vertical and horizontal derivatives and analytic signal, the edge
detectors tilt, theta, THDR of the tilt, NSTD and their normalized vertical derivatives,
with the zero line of the vertical derivative and the map layers over them; and the
regional–residual split the inversion will use. Besides trend surfaces the regional field
can now be an upward continuation, a Butterworth low-pass or a band-pass
(`geoinv3d/methods/regional.py`), computed by the worker on the gridded data. Magnetic
total-field data can also be **inverted reduced to the pole** (the card's *Data inverted*;
`rtp: true` in a dataset): the worker reduces each file's full data, samples it at the
job's stations, takes the regional field from the reduced data and inverts under a vertical
field. It assumes induced magnetization (not with MVI), and the page refuses it below
|I| = 30° and warns below 45°, where the total field is the safer choice.

The page also fetches what a survey needs but its files often lack: **a DEM** for the data
area and the mesh's padding (SRTM 1 arc-second from the public AWS terrain tiles, else
ETOPO 2022 from NOAA; cached, resampled into the data's coordinates; `geoinv3d/io/dem.py`)
and **the inducing field** from IGRF-14 at the centre of the magnetic data on the survey
date, with the declination from the grid north of the data's coordinates
(`geoinv3d/methods/igrf.py`). The Review step lists **checks before running** (data in the
window, cells against the data spacing, core depth and padding against the survey width,
DEM coverage and relief, the magnetic field, noise floors, memory, AWS credentials), with a
fix button where one applies; errors block the submit. **Save this setup** writes every
setting as JSON (its `params` are what the worker runs), and a job's **Inputs (.zip)** runs
anywhere with `python -m geoinv3d.cloud.worker --local params.json data out`.

## How the DAG Works

The DAG pattern is adapted from [jif3d_visualization](https://git.tu-berlin.de/applied-geophysics/jif3d_visualization.git). The core idea:

1. **Everything is a node.** Each operation (load data, create mesh, set parameters, run forward model, run inversion) becomes a `Node` in a `Graph`.

2. **Nodes are pure functions.** `_compute(inputs)` takes its upstream nodes' outputs and produces its own output. Deterministic, no side effects.

3. **Parameters are serializable.** `params()` returns a JSON-compatible dict. `from_params()` rebuilds the node. This means the entire workflow can be saved and restored.

4. **Dirty propagation.** Changing a parameter calls `invalidate()`, which marks the node and all downstream dependents as dirty. The next `evaluate()` recomputes only what changed.

5. **Provenance.** Every export can write a `.provenance.json` sidecar containing the full lineage: which nodes, with which parameters, produced the result.

## Development Log

See [LOGBOOK.md](LOGBOOK.md) for the development history and design decisions.

## License

MIT
