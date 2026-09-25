"""End-to-end synthetic gravity inversion example.

Creates a synthetic density model with an anomaly, generates forward
data using SimPEG, adds noise, then inverts it — all tracked in the DAG.

Requires: SimPEG, discretize, numpy, matplotlib
Run with: python examples/synthetic_gravity_inversion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geoinv3d.core import Graph
from geoinv3d.core.serialize import save_workflow, graph_to_dict
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.model import PhysicalModel, PhysicalProperty
from geoinv3d.nodes.input_nodes import (
    MeshCreateNode, ModelFromArrayNode, SurveyCreateNode,
)
from geoinv3d.nodes.forward_nodes import ForwardNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode
from geoinv3d.nodes.inversion_nodes import SingleInversionNode
from geoinv3d.viz.dag_view import dag_to_mermaid


def create_anomaly_model(mesh: Mesh3D) -> np.ndarray:
    """Create a density model with a buried block anomaly."""
    nx, ny, nz = mesh.shape
    values = np.zeros(mesh.n_cells)

    # Place a dense block in the center
    v3d = values.reshape(nx, ny, nz)
    cx, cy = nx // 2, ny // 2
    # Block from cells [cx-2:cx+2, cy-2:cy+2, 2:5] (buried at depth)
    x1, x2 = max(cx - 3, 0), min(cx + 3, nx)
    y1, y2 = max(cy - 3, 0), min(cy + 3, ny)
    z1, z2 = 2, min(5, nz)
    v3d[x1:x2, y1:y2, z1:z2] = 0.5  # +0.5 g/cm^3 anomaly

    return values


def main():
    print("=" * 60)
    print("Synthetic Gravity Inversion (end-to-end)")
    print("=" * 60)

    # ── 1. Build mesh and true model ───────────────────────────
    print("\n[1] Creating mesh and true model...")
    graph = Graph()

    mesh_node = MeshCreateNode(
        nx=15, ny=15, nz=8,
        dx=100.0, dy=100.0, dz=50.0,
        name="Survey Mesh",
    )
    graph.add(mesh_node)
    mesh = mesh_node.evaluate()
    print(f"    Mesh: {mesh.shape}, {mesh.n_cells} cells")

    true_values = create_anomaly_model(mesh)
    true_model_node = ModelFromArrayNode(
        mesh_node, true_values, prop="density", name="True Density",
    )
    graph.add(true_model_node)

    # ── 2. Generate synthetic survey ───────────────────────────
    print("[2] Generating survey stations...")
    n_stations = 64
    np.random.seed(42)
    # Grid of stations at the surface
    xs = np.linspace(200, 1300, 8)
    ys = np.linspace(200, 1300, 8)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_stations)])
    print(f"    Stations: {n_stations} on 8x8 grid")

    # ── 3. Forward modeling ────────────────────────────────────
    print("[3] Running forward modeling (SimPEG gravity)...")
    station_survey = SurveyCreateNode(
        locs, np.zeros(n_stations), np.ones(n_stations),
        method="gravity", name="Station Layout",
    )
    graph.add(station_survey)
    fwd_node = ForwardNode(
        true_model_node, station_survey,
        method_type="gravity",
        name="Forward Gravity",
    )
    graph.add(fwd_node)
    dpred = fwd_node.evaluate()
    print(f"    Predicted data range: [{dpred.min():.4f}, {dpred.max():.4f}]")

    # ── 4. Add noise to create observed data ───────────────────
    print("[4] Adding 2% Gaussian noise...")
    noise_level = 0.02
    noise = noise_level * np.abs(dpred) * np.random.randn(n_stations)
    dobs = dpred + noise
    std = noise_level * np.abs(dpred) + 1e-6

    survey_node = SurveyCreateNode(
        locs, dobs, std, method="gravity", name="Observed Gravity",
    )
    graph.add(survey_node)

    # ── 5. Set up inversion ────────────────────────────────────
    print("[5] Setting up inversion...")

    # Starting model: zero everywhere
    start_model_node = ModelFromArrayNode(
        mesh_node, np.zeros(mesh.n_cells), prop="density",
        name="Starting Model",
    )
    graph.add(start_model_node)

    reg_node = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Smoothness Reg",
    )
    graph.add(reg_node)

    inv_node = SingleInversionNode(
        start_model_node, survey_node, reg_node,
        method_type="gravity",
        max_iter=10,
        beta0_ratio=1.0,
        cooling_factor=2.0,
        name="Gravity Inversion",
    )
    graph.add(inv_node)

    # ── 6. Run inversion ───────────────────────────────────────
    print("[6] Running inversion (this may take a minute)...")
    result = inv_node.evaluate()

    print(f"    Iterations: {result.n_iterations}")
    print(f"    Converged: {result.converged}")
    if result.n_iterations > 0:
        print(f"    Final phi_d: {result.phi_d_history[-1]:.2f}")
        print(f"    Final phi_m: {result.phi_m_history[-1]:.4f}")
        if result.final_model is not None:
            rec = result.final_model.values
            print(f"    Recovered model range: [{rec.min():.4f}, {rec.max():.4f}]")
            print(f"    True model range:      [{true_values.min():.4f}, {true_values.max():.4f}]")

    # ── 7. Save workflow (with process data) ────────────────────
    print("\n[7] Saving workflow DAG with process data...")
    doc = graph_to_dict(graph, include_outputs=True)
    print(f"    Total DAG nodes: {len(doc['nodes'])}")
    for n in doc["nodes"]:
        p = n.get("params", {})
        key_params = {k: v for k, v in p.items()
                      if k in ("nx", "method_type", "max_iter", "alpha_s",
                               "value", "factor", "weight")}
        has_output = " [+output]" if "output" in n else ""
        print(f"    #{n['id']:2d} {n['type']:25s} {n['name']:20s} {key_params}{has_output}")

    save_workflow(graph, "synthetic_gravity.geoinv3d.json", include_outputs=True)
    print(f"    Saved to: synthetic_gravity.geoinv3d.json")

    # ── 8. Show Mermaid diagram ────────────────────────────────
    print("\n[8] Mermaid diagram:")
    print(dag_to_mermaid(graph))

    # ── 9. Demonstrate parameter re-tuning ─────────────────────
    print("\n[9] Demonstrating parameter re-tuning...")
    print(f"    Changing alpha_s from 1e-4 to 1e-2...")
    reg_node.set_alphas(alpha_s=1e-2)
    print(f"    Inversion node dirty: {inv_node._dirty}")
    print(f"    (Re-running would recompute with new regularization)")

    print("\n" + "=" * 60)
    print("Done! Full inversion DAG saved with all parameters.")
    print("=" * 60)


if __name__ == "__main__":
    main()
