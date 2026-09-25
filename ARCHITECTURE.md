# Architecture

How the code is organised, for anyone picking it up. This file is the map;
`README.md` is the overview; `LOGBOOK.md` is the chronological record.

## The one-sentence version

> Every step in a geophysical inversion — from mesh creation to parameter
> tuning to export — is a **node in a DAG**; SimPEG does the physics;
> the DAG does the bookkeeping.

## Layering

```
  viz/           matplotlib, PyVista, networkx rendering
   │             reads node outputs, never modifies the graph
  nodes/         DAG node types: input, transform, forward, inversion, export
   │             each node wraps a method or datamodel operation
  methods/       SimPEG wrappers: Gravity, Magnetics, DC, Joint
   │             knows SimPEG API, builds Simulations and DataMisfits
  datamodel/     immutable payloads: Mesh3D, PhysicalModel, SurveyData
   │             pure NumPy — no SimPEG, no VTK at import time
  core/          DAG engine: Node, Graph, serialize
   │             the foundation — everything above depends on this
  io/            file persistence (NumPy, JSON)
```

Dependencies point **downward only**.

## Core: the DAG engine (`core/`)

### Node (`core/node.py`)

The `Node` base class is generic over its output type `T`. Every subclass
implements one method:

```python
def _compute(self, inputs: list[Any]) -> T:
    """Pure function: inputs' outputs → this node's output."""
```

Plus serialization:
- `params()` — JSON-native dict of this node's parameters (no inputs)
- `from_params(params, inputs)` — rebuild from a saved workflow

Key mechanisms:
- **Caching**: `evaluate()` computes once and caches. Subsequent calls
  return the cached value until `invalidate()` is called.
- **Dirty propagation**: `invalidate()` marks this node and its entire
  downstream subtree as dirty. The walk is iterative, not recursive.
- **Iterative evaluation**: `evaluate()` uses an explicit post-order walk
  (stack-based) to avoid stack overflow on long chains.
- **Registry**: `@register_node` adds a class to `NODE_REGISTRY`, making
  it deserializable from saved workflows.

### Graph (`core/node.py`)

A container of nodes with traversal helpers:
- `add(node)` — registers node and its transitive inputs
- `remove(node)` — removes a leaf only (DAG consistency)
- `remove_cascade(node)` — removes node and all downstream dependents
- `lineage(node)` — all ancestors transitively
- `closure(node)` — all descendants transitively

### Serialization (`core/serialize.py`)

- `graph_to_dict(graph)` → `{version, nodes:[{id, type, name, params, inputs}]}`
- `graph_from_dict(doc)` → `(Graph, id_map)` via Kahn's topological sort
- `write_provenance(node, graph, path)` → `.provenance.json` sidecar
- `save_workflow` / `load_workflow` — full DAG persistence

## Data model (`datamodel/`)

All payloads are **frozen dataclasses** with read-only NumPy arrays.

| Class              | Purpose                                          |
|--------------------|--------------------------------------------------|
| `Mesh3D`           | Rectilinear tensor mesh (hx, hy, hz, origin)     |
| `PhysicalModel`    | Values on a mesh + property type + units          |
| `SurveyData`       | Station locations + observed data + uncertainties |
| `InversionResult`  | Iteration history (phi_d, phi_m, beta per step)   |
| `IterationSnapshot`| One iteration's frozen state                      |

`Mesh3D.to_discretize()` converts to a `discretize.TensorMesh` when
SimPEG is needed; `from_discretize()` goes the other way. The datamodel
itself has no SimPEG dependency.

## Methods (`methods/`)

Each method wraps SimPEG's API into a clean interface:

```python
class GravityMethod(MethodBase):
    def make_simulation(mesh, survey) -> Simulation3DIntegral
    def forward(model, survey) -> predicted_data
    def make_dmis(survey, simulation) -> L2DataMisfit
    def make_reg(mesh) -> WeightedLeastSquares
```

| Method               | SimPEG module                        | Property       |
|----------------------|--------------------------------------|----------------|
| `GravityMethod`      | `potential_fields.gravity`           | density        |
| `MagneticsMethod`    | `potential_fields.magnetics`         | susceptibility |
| `DCResistivityMethod`| `electromagnetics.static.resistivity`| conductivity   |
| `JointInversion`     | combines multiple DataMisfits        | multi-property |

## Nodes (`nodes/`)

Each node type follows the `Node` contract. Categories:

| Module                    | Node Types                                                |
|---------------------------|-----------------------------------------------------------|
| `input_nodes.py`          | `MeshCreateNode`, `ModelCreateNode`, `ModelFromArrayNode`, `SurveyCreateNode` |
| `transform_nodes.py`      | `LogTransformNode`, `ScaleNode`, `OffsetNode`, `ConductivityToResistivityNode` |
| `forward_nodes.py`        | `ForwardNode`                                             |
| `regularization_nodes.py` | `RegularizationNode`, `CrossGradientNode`                 |
| `inversion_nodes.py`      | `SingleInversionNode`, `JointInversionNode`               |
| `output_nodes.py`         | `ModelExportNode`, `ResultExportNode`                     |

### Adding a node type — checklist

1. Implement `_compute`, `params`, `from_params`
2. Decorate with `@register_node`
3. Import in `nodes/__init__.py` so it registers on package load
4. Add a serialization round-trip test

## Visualization (`viz/`)

| Function           | Purpose                              | Requires    |
|--------------------|--------------------------------------|-------------|
| `plot_dag`         | DAG graph with networkx + matplotlib | networkx    |
| `dag_to_mermaid`   | Mermaid flowchart text               | (none)      |
| `plot_model_slice` | 2D cross-section through a model     | matplotlib  |
| `plot_convergence` | phi_d / phi_m convergence curves     | matplotlib  |
| `plot_model_3d`    | Interactive 3D volume rendering      | pyvista     |

## Invariants

1. **The graph is the system of record.** If it's not in the DAG, it didn't happen.
2. **Data is immutable.** Frozen dataclasses, read-only arrays. Transforms return new objects.
3. **Nodes are pure functions.** `_compute` is deterministic, no side effects beyond output.
4. **Parameters are JSON-serializable.** Every node can be saved and rebuilt.
5. **Dirty propagation is downward.** Changing a parameter invalidates only downstream.
6. **SimPEG is a backend, not a dependency of the core.** `core/` and `datamodel/` never import SimPEG.

## Data flow in a joint inversion

```
MeshCreateNode ──→ ModelCreateNode (density) ──→ ForwardNode (gravity) ──→ SingleInversionNode
                                                                             ↑
                   SurveyCreateNode (gravity) ────────────────────────────────┘
                                                                             ↑
                   RegularizationNode (alpha_s, alpha_x, ...) ───────────────┘

MeshCreateNode ──→ ModelCreateNode (suscept.) ──→ ForwardNode (magnetics) ──→ SingleInversionNode
                                                                              ↑
                   SurveyCreateNode (magnetics) ──────────────────────────────┘

                                         both ──→ JointInversionNode ──→ ResultExportNode
```
