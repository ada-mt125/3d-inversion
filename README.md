# GeoInv3D

**DAG-based 3D Geophysical Joint Inversion Framework**

A Python framework for 3D geophysical joint inversion that uses a Directed Acyclic Graph (DAG) to track every parameter change, making inversion workflows fully reproducible and inspectable.

## Key Features

- **DAG-based workflow tracking** — Every step (mesh creation, model setup, forward modeling, inversion, parameter tuning) is a node in a DAG. Every parameter change is recorded automatically.
- **SimPEG backend** — Uses [SimPEG](https://simpeg.xyz/) for forward modeling and inversion, supporting gravity, magnetics, DC resistivity, and joint inversion.
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
