"""Generate a complete workflow JSON with synthetic inversion results.

Creates a self-contained demo for the unified viewer: DAG workflow,
convergence curves, data fit, depth slices, and cross-sections.

No SimPEG needed — uses analytic forward model and mock inversion results.

Usage:
    python examples/generate_workflow_demo.py
    python examples/generate_workflow_demo.py --open
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

G_CONST = 6.674e-3  # G * 1e5 * 1e3 for density in g/cm³, distances in m, output in mGal


def forward_gravity(x_sta, y_sta, x_c, y_c, z_c, dV, model):
    """Vectorized gravity forward using point-mass approximation."""
    active = np.abs(model) > 1e-10
    cx, cy, cz, rho = x_c[active], y_c[active], z_c[active], model[active]
    dx = x_sta[:, None] - cx[None, :]
    dy = y_sta[:, None] - cy[None, :]
    dz = -cz[None, :]  # stations at z=0
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 1.0)
    return -G_CONST * np.sum(rho[None, :] * dV * dz / r**3, axis=1)


def make_convergence(n_iter, n_data):
    """Generate realistic convergence curves."""
    iters = []
    phi_d = n_data * 40
    beta = 8.0
    for i in range(n_iter):
        phi_d = max(n_data * 0.85, phi_d * 0.52)
        phi_m = 0.002 + 0.06 * (1 - math.exp(-i / 3))
        if i > 5:
            beta *= 0.65
            phi_m *= 1.15
        iters.append({
            "iteration": i + 1,
            "phi_d": round(phi_d, 3),
            "phi_m": round(phi_m, 6),
            "phi_total": round(phi_d + beta * phi_m, 3),
            "beta": round(beta, 4),
        })
    return iters


def main():
    np.random.seed(42)

    # ── Mesh ──
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 50.0, 50.0, 25.0

    x_c = dx * (np.arange(nx) + 0.5)
    y_c = dy * (np.arange(ny) + 0.5)
    z_c = -dz * (np.arange(nz) + 0.5)  # negative = depth
    x_e = (dx * np.arange(nx + 1)).tolist()
    y_e = (dy * np.arange(ny + 1)).tolist()
    z_e = (-dz * np.arange(nz + 1)).tolist()

    n_cells = nx * ny * nz
    # Build cell center coordinate arrays (flat, index = ix + iy*nx + iz*nx*ny)
    cell_x = np.zeros(n_cells)
    cell_y = np.zeros(n_cells)
    cell_z = np.zeros(n_cells)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                idx = ix + iy * nx + iz * nx * ny
                cell_x[idx] = x_c[ix]
                cell_y[idx] = y_c[iy]
                cell_z[idx] = z_c[iz]

    # ── True model: spherical density anomaly ──
    true_model = np.zeros(n_cells)
    cx, cy, cz = 500.0, 500.0, -125.0
    radius = 175.0
    for i in range(n_cells):
        d = math.sqrt((cell_x[i] - cx)**2 + (cell_y[i] - cy)**2 + (cell_z[i] - cz)**2)
        if d < radius:
            true_model[i] = 0.4 * (1.0 - (d / radius)**2)

    # ── Recovered model: smoothed true + slight underestimation ──
    vol = true_model.reshape((nz, ny, nx))  # [iz][iy][ix]
    rec = vol.copy()
    for _ in range(4):
        p = np.pad(rec, 1, mode="constant")
        rec = (p[1:-1, 1:-1, 1:-1] * 6 +
               p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +
               p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +
               p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:]) / 12
    rec = rec * 0.80 + np.random.randn(nz, ny, nx) * 0.003
    rec = np.clip(rec, -0.02, 0.45)
    recovered = rec.ravel()  # back to flat, same convention

    # ── Stations ──
    xs = np.linspace(50, 950, 12)
    ys = np.linspace(50, 950, 12)
    xx, yy = np.meshgrid(xs, ys)
    x_sta, y_sta = xx.ravel(), yy.ravel()
    n_sta = len(x_sta)

    # ── Forward gravity ──
    dV = dx * dy * dz
    print("Forward gravity (true model)...")
    g_true = forward_gravity(x_sta, y_sta, cell_x, cell_y, cell_z, dV, true_model)
    noise = 0.02 * np.abs(g_true) * np.random.randn(n_sta)
    g_obs = g_true + noise

    print("Forward gravity (recovered model)...")
    g_pred = forward_gravity(x_sta, y_sta, cell_x, cell_y, cell_z, dV, recovered)
    residuals = g_obs - g_pred

    # ── Convergence ──
    iterations = make_convergence(15, n_sta)
    for i, it in enumerate(iterations):
        frac = min(1.0, (i + 1) / 12)
        m = true_model * (1 - frac) * 0.3 + recovered * frac
        it["model_min"] = round(float(m.min()), 6)
        it["model_max"] = round(float(m.max()), 6)
        it["model_mean"] = round(float(m.mean()), 6)

    # ── Round helper ──
    def rl(a, d=6):
        return [round(float(v), d) for v in a]

    # ── Build workflow JSON ──
    workflow = {
        "version": 2,
        "created": "2026-09-24T10:00:00Z",
        "nodes": [
            {
                "id": 0, "type": "SurveyCreateNode", "name": "Gravity Survey",
                "params": {
                    "method": "gravity", "n_stations": n_sta,
                    "noise_pct": 0.02, "noise_floor": 0.5,
                    "data_range": f"[{g_obs.min():.2f}, {g_obs.max():.2f}] mGal",
                },
                "inputs": [],
            },
            {
                "id": 1, "type": "MeshCreateNode", "name": "3D Tensor Mesh",
                "params": {
                    "nx": nx, "ny": ny, "nz": nz,
                    "dx": dx, "dy": dy, "dz": dz,
                    "n_cells": n_cells,
                    "x_extent": f"[0, {nx*dx}] m",
                    "y_extent": f"[0, {ny*dy}] m",
                    "z_extent": f"[0, {nz*dz}] m depth",
                },
                "inputs": [],
            },
            {
                "id": 2, "type": "ModelCreateNode", "name": "Starting Model",
                "params": {
                    "value": 0.0, "property": "density", "unit": "g/cm³",
                },
                "inputs": [1],
            },
            {
                "id": 3, "type": "RegularizationNode", "name": "Sparse Regularization",
                "params": {
                    "alpha_s": 0.001, "alpha_x": 1.0,
                    "alpha_y": 1.0, "alpha_z": 1.0,
                },
                "inputs": [],
            },
            {
                "id": 4, "type": "SparseInversionNode", "name": "Gravity Inversion",
                "params": {
                    "method_type": "gravity",
                    "norms": [0, 2, 2, 1],
                    "max_iter": 25,
                    "beta0_ratio": 1.0,
                    "irls_cooling_factor": 1.1,
                    "max_irls_iterations": 12,
                    "use_preconditioner": True,
                },
                "inputs": [0, 2, 3],
                "output": {
                    "type": "InversionResult",
                    "method": "gravity",
                    "converged": True,
                    "n_iterations": len(iterations),
                    "iterations": iterations,
                    "final_model": {
                        "prop": "density", "unit": "g/cm³",
                        "n_cells": n_cells,
                        "min": round(float(recovered.min()), 6),
                        "max": round(float(recovered.max()), 6),
                        "mean": round(float(recovered.mean()), 6),
                    },
                    "model_3d": {
                        "nx": nx, "ny": ny, "nz": nz,
                        "x_edges": x_e, "y_edges": y_e, "z_edges": z_e,
                        "values": rl(recovered),
                        "true_values": rl(true_model),
                    },
                    "data_fit": {
                        "unit": "mGal",
                        "x_stations": rl(x_sta, 1),
                        "y_stations": rl(y_sta, 1),
                        "observed": rl(g_obs, 4),
                        "predicted": rl(g_pred, 4),
                        "residuals": rl(residuals, 4),
                        "rms": round(float(np.sqrt(np.mean(residuals**2))), 4),
                        "chi2": round(float(np.sum((residuals / (0.02 * np.abs(g_obs) + 0.5))**2) / n_sta), 3),
                    },
                },
            },
            {
                "id": 5, "type": "ResultExportNode", "name": "Export VTK + JSON",
                "params": {"format": "VTK + JSON", "include_model": True},
                "inputs": [4],
            },
        ],
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "demo_workflow.geoinv3d.json")
    with open(out_path, "w") as f:
        json.dump(workflow, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved: {out_path} ({size_kb:.1f} KB)")
    print(f"  Nodes: {len(workflow['nodes'])}")
    print(f"  Mesh: {nx}x{ny}x{nz} = {n_cells} cells")
    print(f"  Stations: {n_sta}")
    print(f"  True model:  [{true_model.min():.4f}, {true_model.max():.4f}] g/cm³")
    print(f"  Recovered:   [{recovered.min():.4f}, {recovered.max():.4f}] g/cm³")
    print(f"  Data range:  [{g_obs.min():.2f}, {g_obs.max():.2f}] mGal")
    print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.4f} mGal")

    if "--open" in sys.argv:
        from geoinv3d.viz.serve_dag import generate_viewer
        import webbrowser
        html = generate_viewer(out_path)
        webbrowser.open(f"file:///{os.path.abspath(html)}")
        print(f"  Viewer: {html}")


if __name__ == "__main__":
    main()
