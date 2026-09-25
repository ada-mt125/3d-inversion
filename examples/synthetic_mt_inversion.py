"""Synthetic MT inversion with a conductive anomaly.

Creates a 3D conductivity model with a buried conductive block,
generates synthetic impedance data, and runs an MT inversion.

NOTE: MT in 3D is computationally expensive. This example uses a
very small mesh (5x5x4 = 100 cells) for quick testing. Real MT
inversions require much larger meshes and should use the cloud
module or a machine with more RAM.

Requires: simpeg, discretize, numpy
Run with: python examples/synthetic_mt_inversion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geoinv3d.core import Graph
from geoinv3d.core.serialize import save_workflow, graph_to_dict
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.model import PhysicalModel, PhysicalProperty
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.nodes.input_nodes import MeshCreateNode, ModelFromArrayNode, SurveyCreateNode
from geoinv3d.nodes.forward_nodes import ForwardNode
from geoinv3d.nodes.regularization_nodes import RegularizationNode
from geoinv3d.nodes.inversion_nodes import SingleInversionNode
from geoinv3d.methods.mt import MTMethod


def main():
    np.random.seed(42)
    print("=" * 60)
    print("Synthetic MT Inversion")
    print("=" * 60)

    # ── 1. Mesh ──────────────────────────────────────────────
    print("\n[1] Creating mesh...")
    nx, ny, nz = 5, 5, 4
    dx, dy, dz = 200.0, 200.0, 100.0
    mesh = Mesh3D.uniform(nx, ny, nz, dx, dy, dz)
    print(f"    Mesh: {mesh.shape}, {mesh.n_cells} cells")

    graph = Graph()
    mesh_node = MeshCreateNode(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
        name="MT Mesh",
    )
    graph.add(mesh_node)

    # ── 2. True conductivity model ───────────────────────────
    print("\n[2] Creating true conductivity model...")
    sigma_bg = 0.01  # 100 Ohm.m background
    sigma_anomaly = 0.1  # 10 Ohm.m conductive block

    sigma_true = np.full(mesh.n_cells, sigma_bg)
    s3d = sigma_true.reshape(nx, ny, nz)
    s3d[2:4, 2:4, 1:3] = sigma_anomaly

    # MT uses log-conductivity as model parameter
    log_sigma_true = np.log(sigma_true)
    print(f"    Background: {sigma_bg} S/m ({1/sigma_bg:.0f} Ohm.m)")
    print(f"    Anomaly: {sigma_anomaly} S/m ({1/sigma_anomaly:.0f} Ohm.m)")

    true_model_node = ModelFromArrayNode(
        mesh_node, log_sigma_true,
        prop="density",  # reusing density prop for log-sigma
        name="True log(sigma)",
    )
    graph.add(true_model_node)

    # ── 3. MT forward modeling ───────────────────────────────
    print("\n[3] Forward modeling...")
    frequencies = [0.01, 0.1, 1.0]
    components = ["xy_real", "xy_imag"]
    n_stations = 4

    # Station grid on surface
    xs = np.linspace(300, 700, 2)
    ys = np.linspace(300, 700, 2)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_stations)])

    n_data = n_stations * len(components) * len(frequencies)

    mt = MTMethod(
        frequencies=frequencies,
        components=components,
        sigma_background=sigma_bg,
    )

    # Build SimPEG simulation for forward
    survey_dummy = SurveyData(
        locs, np.zeros(n_data), np.ones(n_data), method="mt",
    )
    true_model = PhysicalModel(mesh, log_sigma_true, PhysicalProperty.DENSITY, "log_sigma")
    dpred = mt.forward(true_model, survey_dummy)
    print(f"    Frequencies: {frequencies} Hz")
    print(f"    Stations: {n_stations}")
    print(f"    Data points: {len(dpred)}")
    print(f"    Data range: [{dpred.min():.6f}, {dpred.max():.6f}]")

    # ── 4. Add noise ─────────────────────────────────────────
    print("\n[4] Adding noise...")
    noise_floor = 1e-4
    std = np.abs(dpred) * 0.05 + noise_floor
    noise = std * np.random.randn(n_data)
    dobs = dpred + noise
    print(f"    5% relative + {noise_floor} floor")

    obs_node = SurveyCreateNode(
        locs, dobs, std, method="mt", name="Observed MT",
    )
    graph.add(obs_node)

    # ── 5. Starting model (homogeneous) ──────────────────────
    print("\n[5] Starting model: uniform {:.0f} Ohm.m".format(1/sigma_bg))
    start_log_sigma = np.full(mesh.n_cells, np.log(sigma_bg))
    start_node = ModelFromArrayNode(
        mesh_node, start_log_sigma, prop="density",
        name="Start log(sigma)",
    )
    graph.add(start_node)

    # ── 6. Inversion ─────────────────────────────────────────
    print("\n[6] Running MT inversion...")
    reg_node = RegularizationNode(
        alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        name="MT Reg",
    )
    graph.add(reg_node)

    inv_node = SingleInversionNode(
        start_node, obs_node, reg_node,
        method_type="mt",
        method_kwargs={
            "frequencies": frequencies,
            "components": components,
            "sigma_background": sigma_bg,
        },
        max_iter=5,
        beta0_ratio=1.0,
        cooling_factor=2.0,
        name="MT Inversion",
    )
    graph.add(inv_node)
    result = inv_node.evaluate()

    print(f"    Iterations: {result.n_iterations}")
    print(f"    Final phi_d: {result.phi_d_history[-1]:.2f}")

    if result.final_model:
        recovered_sigma = np.exp(result.final_model.values)
        print(f"    Recovered sigma: [{recovered_sigma.min():.4f}, "
              f"{recovered_sigma.max():.4f}] S/m")
        print(f"    True anomaly sigma: {sigma_anomaly}")

    # ── 7. Save ──────────────────────────────────────────────
    print("\n[7] Saving workflow...")
    save_workflow(graph, "mt_inversion.geoinv3d.json", include_outputs=True)
    print(f"    Saved: mt_inversion.geoinv3d.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
