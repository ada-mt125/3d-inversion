"""Joint gravity and magnetics inversion with synthetic data.

Creates a buried anomaly with both density and susceptibility contrasts,
generates synthetic data for both methods, then runs:
  1. Individual gravity inversion
  2. Individual magnetics inversion
  3. Joint gravity + magnetics inversion (jif3d-style with Wires projections)

All tracked in the DAG with full process data.

Requires: simpeg, discretize, numpy
Run with: python examples/joint_gravity_magnetics.py
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
from geoinv3d.nodes.inversion_nodes import SingleInversionNode, JointInversionNode
from geoinv3d.viz.dag_view import dag_to_mermaid


def create_anomaly(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:
    """Create a buried block with density and susceptibility contrasts."""
    nx, ny, nz = mesh.shape
    density = np.zeros(mesh.n_cells)
    suscept = np.zeros(mesh.n_cells)

    d3d = density.reshape(nx, ny, nz)
    s3d = suscept.reshape(nx, ny, nz)

    cx, cy = nx // 2, ny // 2
    x1, x2 = max(cx - 3, 0), min(cx + 3, nx)
    y1, y2 = max(cy - 3, 0), min(cy + 3, ny)
    z1, z2 = 2, min(5, nz)

    d3d[x1:x2, y1:y2, z1:z2] = 0.5    # +0.5 g/cm^3
    s3d[x1:x2, y1:y2, z1:z2] = 0.05   # +0.05 SI susceptibility

    return density, suscept


def create_stations(n_side: int = 8) -> np.ndarray:
    """Create a regular grid of surface stations."""
    xs = np.linspace(200, 1300, n_side)
    ys = np.linspace(200, 1300, n_side)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_side**2)])


def main():
    np.random.seed(42)
    print("=" * 70)
    print("Joint Gravity + Magnetics Inversion")
    print("=" * 70)

    # ── 1. Shared mesh and true models ────────────────────────
    print("\n[1] Creating mesh and true models...")
    graph = Graph()

    mesh_node = MeshCreateNode(
        nx=15, ny=15, nz=8,
        dx=100.0, dy=100.0, dz=50.0,
        name="Survey Mesh",
    )
    graph.add(mesh_node)
    mesh = mesh_node.evaluate()
    print(f"    Mesh: {mesh.shape}, {mesh.n_cells} cells")

    true_density, true_suscept = create_anomaly(mesh)

    true_density_node = ModelFromArrayNode(
        mesh_node, true_density, prop="density", name="True Density",
    )
    graph.add(true_density_node)
    print(f"    Density anomaly: max={true_density.max():.2f} g/cm3")

    true_suscept_node = ModelFromArrayNode(
        mesh_node, true_suscept, prop="susceptibility", name="True Susceptibility",
    )
    graph.add(true_suscept_node)
    print(f"    Susceptibility anomaly: max={true_suscept.max():.3f} SI")

    # ── 2. Survey stations ────────────────────────────────────
    print("\n[2] Creating survey stations...")
    locs = create_stations(8)
    n_stations = len(locs)
    print(f"    {n_stations} stations on 8x8 grid")

    # Station layout nodes for forward modeling
    grav_stations = SurveyCreateNode(
        locs, np.zeros(n_stations), np.ones(n_stations),
        method="gravity", name="Gravity Stations",
    )
    graph.add(grav_stations)

    mag_stations = SurveyCreateNode(
        locs, np.zeros(n_stations), np.ones(n_stations),
        method="magnetics", name="Magnetics Stations",
    )
    graph.add(mag_stations)

    # ── 3. Forward modeling ───────────────────────────────────
    print("\n[3] Forward modeling...")

    grav_fwd = ForwardNode(
        true_density_node, grav_stations,
        method_type="gravity",
        name="Gravity Forward",
    )
    graph.add(grav_fwd)
    grav_pred = grav_fwd.evaluate()
    print(f"    Gravity: [{grav_pred.min():.4f}, {grav_pred.max():.4f}] mGal")

    mag_fwd = ForwardNode(
        true_suscept_node, mag_stations,
        method_type="magnetics",
        method_kwargs={"inducing_field": (50000.0, 70.0, 0.0)},
        name="Magnetics Forward",
    )
    graph.add(mag_fwd)
    mag_pred = mag_fwd.evaluate()
    print(f"    Magnetics: [{mag_pred.min():.2f}, {mag_pred.max():.2f}] nT")

    # ── 4. Add noise to create observed data ──────────────────
    print("\n[4] Adding noise...")
    noise_pct = 0.02

    grav_noise = noise_pct * np.abs(grav_pred) * np.random.randn(n_stations)
    grav_dobs = grav_pred + grav_noise
    grav_std = noise_pct * np.abs(grav_pred) + 1e-6

    grav_obs_node = SurveyCreateNode(
        locs, grav_dobs, grav_std,
        method="gravity", name="Observed Gravity",
    )
    graph.add(grav_obs_node)
    print(f"    Gravity: 2% Gaussian noise added")

    mag_noise = noise_pct * np.abs(mag_pred) * np.random.randn(n_stations)
    mag_dobs = mag_pred + mag_noise
    mag_std = noise_pct * np.abs(mag_pred) + 1e-6

    mag_obs_node = SurveyCreateNode(
        locs, mag_dobs, mag_std,
        method="magnetics", name="Observed Magnetics",
    )
    graph.add(mag_obs_node)
    print(f"    Magnetics: 2% Gaussian noise added")

    # ── 5. Starting models (zero) ─────────────────────────────
    print("\n[5] Creating starting models...")
    start_density = ModelFromArrayNode(
        mesh_node, np.zeros(mesh.n_cells), prop="density",
        name="Start Density",
    )
    graph.add(start_density)

    start_suscept = ModelFromArrayNode(
        mesh_node, np.zeros(mesh.n_cells), prop="susceptibility",
        name="Start Susceptibility",
    )
    graph.add(start_suscept)

    # ── 6. Individual gravity inversion ───────────────────────
    print("\n[6] Running gravity-only inversion...")
    grav_reg = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Gravity Reg",
    )
    graph.add(grav_reg)

    grav_inv = SingleInversionNode(
        start_density, grav_obs_node, grav_reg,
        method_type="gravity",
        max_iter=10, beta0_ratio=1.0, cooling_factor=2.0,
        name="Gravity Inversion",
    )
    graph.add(grav_inv)
    grav_result = grav_inv.evaluate()

    print(f"    Iterations: {grav_result.n_iterations}, "
          f"converged: {grav_result.converged}")
    if grav_result.final_model:
        rec = grav_result.final_model.values
        print(f"    Recovered density: [{rec.min():.4f}, {rec.max():.4f}]")
    print(f"    Final phi_d: {grav_result.phi_d_history[-1]:.2f}")

    # ── 7. Individual magnetics inversion ─────────────────────
    print("\n[7] Running magnetics-only inversion...")
    mag_reg = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Magnetics Reg",
    )
    graph.add(mag_reg)

    mag_inv = SingleInversionNode(
        start_suscept, mag_obs_node, mag_reg,
        method_type="magnetics",
        method_kwargs={"inducing_field": (50000.0, 70.0, 0.0)},
        max_iter=15, beta0_ratio=1e-2, cooling_factor=2.0,
        name="Magnetics Inversion",
    )
    graph.add(mag_inv)
    mag_result = mag_inv.evaluate()

    print(f"    Iterations: {mag_result.n_iterations}, "
          f"converged: {mag_result.converged}")
    if mag_result.final_model:
        rec = mag_result.final_model.values
        print(f"    Recovered susceptibility: [{rec.min():.5f}, {rec.max():.5f}]")
    print(f"    Final phi_d: {mag_result.phi_d_history[-1]:.2f}")

    # ── 8. Joint gravity + magnetics inversion ────────────────
    print("\n[8] Running joint gravity + magnetics inversion...")
    joint_reg_grav = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Joint Grav Reg",
    )
    graph.add(joint_reg_grav)

    joint_reg_mag = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="Joint Mag Reg",
    )
    graph.add(joint_reg_mag)

    joint_inv = JointInversionNode(
        method_types=["gravity", "magnetics"],
        method_kwargs_list=[
            {},
            {"inducing_field": (50000.0, 70.0, 0.0)},
        ],
        weights=[1.0, 1.0],
        max_iter=20,
        beta0_ratio=1e-2,
        cooling_factor=2.0,
        name="Joint Grav+Mag Inversion",
        inputs=[
            start_density, grav_obs_node, joint_reg_grav,
            start_suscept, mag_obs_node, joint_reg_mag,
        ],
    )
    graph.add(joint_inv)
    joint_result = joint_inv.evaluate()

    print(f"    Iterations: {joint_result.n_iterations}, "
          f"converged: {joint_result.converged}")
    print(f"    Final phi_d: {joint_result.phi_d_history[-1]:.2f}")
    if "gravity" in joint_result.recovered_models:
        gj = joint_result.recovered_models["gravity"]
        print(f"    Joint density: [{gj.min():.4f}, {gj.max():.4f}]")
    if "magnetics" in joint_result.recovered_models:
        mj = joint_result.recovered_models["magnetics"]
        print(f"    Joint susceptibility: [{mj.min():.5f}, {mj.max():.5f}]")

    # ── 8b. Joint inversion WITH cross-gradient coupling ───────
    print("\n[8b] Running joint inversion with cross-gradient coupling...")
    cg_reg_grav = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="CG Grav Reg",
    )
    graph.add(cg_reg_grav)

    cg_reg_mag = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="CG Mag Reg",
    )
    graph.add(cg_reg_mag)

    cg_inv = JointInversionNode(
        method_types=["gravity", "magnetics"],
        method_kwargs_list=[
            {},
            {"inducing_field": (50000.0, 70.0, 0.0)},
        ],
        weights=[1.0, 1.0],
        max_iter=20,
        beta0_ratio=1e-2,
        cooling_factor=2.0,
        cross_gradient_weight=10.0,
        name="Joint+CG Inversion",
        inputs=[
            start_density, grav_obs_node, cg_reg_grav,
            start_suscept, mag_obs_node, cg_reg_mag,
        ],
    )
    graph.add(cg_inv)
    cg_result = cg_inv.evaluate()

    print(f"    Iterations: {cg_result.n_iterations}, "
          f"converged: {cg_result.converged}")
    print(f"    Final phi_d: {cg_result.phi_d_history[-1]:.2f}")
    if "gravity" in cg_result.recovered_models:
        cg_grav = cg_result.recovered_models["gravity"]
        print(f"    CG density: [{cg_grav.min():.4f}, {cg_grav.max():.4f}]")
    if "magnetics" in cg_result.recovered_models:
        cg_mag = cg_result.recovered_models["magnetics"]
        print(f"    CG susceptibility: [{cg_mag.min():.5f}, {cg_mag.max():.5f}]")

    # ── 9. Save workflow with full process data ───────────────
    print("\n[9] Saving workflow DAG with process data...")
    doc = graph_to_dict(graph, include_outputs=True)
    print(f"    Total DAG nodes: {len(doc['nodes'])}")
    for n in doc["nodes"]:
        has_out = " [+output]" if "output" in n else ""
        print(f"    #{n['id']:2d} {n['type']:25s} {n['name']}{has_out}")

    save_workflow(graph, "joint_grav_mag.geoinv3d.json", include_outputs=True)
    print(f"    Saved: joint_grav_mag.geoinv3d.json")

    # ── 10. Generate viewer ───────────────────────────────────
    print("\n[10] Generating interactive viewer...")
    from geoinv3d.viz import generate_viewer
    out = generate_viewer("joint_grav_mag.geoinv3d.json")
    print(f"    Generated: {out}")

    # ── 11. Summary comparison ────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Model Recovery Comparison")
    print("=" * 70)
    print(f"{'':20s} {'True':>12s} {'Gravity':>12s} {'Magnetics':>12s} {'Joint':>12s} {'Joint+CG':>12s}")
    print("-" * 82)

    true_d_max = true_density.max()
    grav_d_max = grav_result.final_model.values.max() if grav_result.final_model else 0
    joint_d_max = joint_result.recovered_models.get("gravity", np.array([0])).max()
    cg_d_max = cg_result.recovered_models.get("gravity", np.array([0])).max()
    print(f"{'Density max':20s} {true_d_max:12.4f} {grav_d_max:12.4f} {'  -':>12s} {joint_d_max:12.4f} {cg_d_max:12.4f}")

    true_s_max = true_suscept.max()
    mag_s_max = mag_result.final_model.values.max() if mag_result.final_model else 0
    joint_s_max = joint_result.recovered_models.get("magnetics", np.array([0])).max()
    cg_s_max = cg_result.recovered_models.get("magnetics", np.array([0])).max()
    print(f"{'Suscept max':20s} {true_s_max:12.5f} {'  -':>12s} {mag_s_max:12.5f} {joint_s_max:12.5f} {cg_s_max:12.5f}")

    print(f"{'Grav phi_d':20s} {'':>12s} {grav_result.phi_d_history[-1]:12.2f} {'':>12s} {'':>12s} {'':>12s}")
    print(f"{'Mag phi_d':20s} {'':>12s} {'':>12s} {mag_result.phi_d_history[-1]:12.2f} {'':>12s} {'':>12s}")
    print(f"{'Joint phi_d':20s} {'':>12s} {'':>12s} {'':>12s} {joint_result.phi_d_history[-1]:12.2f} {'':>12s}")
    print(f"{'Joint+CG phi_d':20s} {'':>12s} {'':>12s} {'':>12s} {'':>12s} {cg_result.phi_d_history[-1]:12.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
