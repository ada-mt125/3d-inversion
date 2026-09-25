"""Optimized gravity + magnetics inversion with Sparse (IRLS) regularization.

Following the SimPEG user-tutorial approach (inv-gravity-anomaly-3d):
  1. Padded TensorMesh — expanding cells avoid boundary artifacts
  2. Sparse regularization (L0 smallness + L1 gradients) → compact models
  3. UpdateSensitivityWeights → counteracts depth decay
  4. ProjectedGNCG → enforces model bounds
  5. alpha_s = dh**-2 → cell-size dependent smallness scaling

Run: python examples/optimized_inversion_with_viz.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.model import PhysicalModel, PhysicalProperty
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.gravity import GravityMethod
from geoinv3d.methods.magnetics import MagneticsMethod
from geoinv3d.methods.directives import IterationCollector
from geoinv3d.datamodel.result import InversionResult, IterationSnapshot
from geoinv3d.viz.model3d_viewer import generate_model3d_viewer

# ── Configuration ───────────────────────────────────────────
NX_CORE, NY_CORE, NZ_CORE = 30, 30, 15
DX, DY, DZ = 50.0, 50.0, 25.0
N_PAD = 5
PAD_FACTOR = 1.3

BLOCK_DENSITY = 0.5
BLOCK_SUSCEPT = 0.05


def create_padded_mesh():
    """Mesh with expanding padding cells around a uniform core.

    Core: NX_CORE × NY_CORE × NZ_CORE uniform cells.
    Padding: N_PAD expanding cells on each horizontal side and at the bottom.
    Origin shifted so core domain starts at (0, 0, 0).
    """
    hx_pad = DX * PAD_FACTOR ** np.arange(1, N_PAD + 1)
    hy_pad = DY * PAD_FACTOR ** np.arange(1, N_PAD + 1)
    hz_pad = DZ * PAD_FACTOR ** np.arange(1, N_PAD + 1)

    hx = np.r_[hx_pad[::-1], np.full(NX_CORE, DX), hx_pad]
    hy = np.r_[hy_pad[::-1], np.full(NY_CORE, DY), hy_pad]
    hz = np.r_[np.full(NZ_CORE, DZ), hz_pad]

    origin = (-hx_pad.sum(), -hy_pad.sum(), 0.0)
    return Mesh3D(hx=hx, hy=hy, hz=hz, origin=origin)


def create_anomaly(mesh):
    """Buried block anomaly using cell-center coordinates (order-safe)."""
    dmesh = mesh.to_discretize()
    cc = dmesh.cell_centers

    x_center = NX_CORE * DX / 2
    y_center = NY_CORE * DY / 2
    bx_half = max(NX_CORE // 5, 2) * DX
    by_half = max(NY_CORE // 5, 2) * DY
    z_top = 2 * DZ
    z_bot = z_top + max(NZ_CORE // 4, 2) * DZ

    mask = (
        (cc[:, 0] >= x_center - bx_half) & (cc[:, 0] <= x_center + bx_half) &
        (cc[:, 1] >= y_center - by_half) & (cc[:, 1] <= y_center + by_half) &
        (cc[:, 2] >= z_top) & (cc[:, 2] <= z_bot)
    )

    density = np.zeros(mesh.n_cells)
    suscept = np.zeros(mesh.n_cells)
    density[mask] = BLOCK_DENSITY
    suscept[mask] = BLOCK_SUSCEPT

    n_block = mask.sum()
    print(f"    Block: {n_block} cells, density={BLOCK_DENSITY}, suscept={BLOCK_SUSCEPT}")
    print(f"    Block extent: x=[{x_center-bx_half:.0f},{x_center+bx_half:.0f}], "
          f"y=[{y_center-by_half:.0f},{y_center+by_half:.0f}], "
          f"z=[{z_top:.0f},{z_bot:.0f}] m")

    return density, suscept


def run_sparse_inversion(
    method, mesh, survey, m0,
    max_irls_iterations=40,
    beta0_ratio=10.0,
    norms=None,
    alpha_s=None,
    upper_bound=1.0,
    lower_bound=-1.0,
):
    """Sparse IRLS inversion following the SimPEG tutorial approach."""
    from simpeg import (
        optimization, inverse_problem, inversion,
        directives, regularization,
    )

    if norms is None:
        norms = [0, 1, 1, 1]

    sim = method.make_simulation_full(mesh, survey)
    dmis = method.make_dmis(survey, sim)
    dmesh = mesh.to_discretize()

    dh = min(dmesh.h[0].min(), dmesh.h[1].min(), dmesh.h[2].min())
    if alpha_s is None:
        alpha_s = dh**-2

    reference_model = np.zeros(dmesh.n_cells)

    reg = regularization.Sparse(
        dmesh,
        alpha_s=alpha_s,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        norms=norms,
        reference_model=reference_model,
        reference_model_in_smooth=False,
    )

    opt = optimization.ProjectedGNCG(
        maxIter=100,
        lower=lower_bound,
        upper=upper_bound,
        maxIterLS=20,
        cg_maxiter=20,
        cg_rtol=1e-2,
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    collector = IterationCollector()

    update_irls = directives.UpdateIRLS(
        cooling_factor=2,
        cooling_rate=1,
        chifact_start=1.0,
        f_min_change=1e-4,
        max_irls_iterations=max_irls_iterations,
    )
    sens_weight = directives.UpdateSensitivityWeights(every_iteration=False)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)

    directive_list = [
        collector,
        update_irls,
        sens_weight,
        starting_beta,
        update_jacobi,
    ]

    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

    print(f"    norms={norms}, alpha_s={alpha_s:.4f}, beta0_ratio={beta0_ratio}")
    print(f"    bounds=[{lower_bound}, {upper_bound}], m0 range=[{m0.min():.1e}, {m0.max():.1e}]")
    m_rec = inv.run(m0)

    result = InversionResult(method=method.method_name, converged=True)
    for snap in collector.snapshots:
        result.add_iteration(snap)
    result.final_model = PhysicalModel(mesh, m_rec, PhysicalProperty.DENSITY, "recovered")

    dpred = sim.dpred(m_rec)
    return result, m_rec, dpred


def run_l2_inversion(
    method, mesh, survey, m0,
    max_iter=100,
    beta0_ratio=10.0,
    cooling_factor=2.0,
    alpha_s=None,
):
    """L2 inversion with sensitivity weighting (tutorial approach)."""
    from simpeg import (
        optimization, inverse_problem, inversion,
        directives, regularization,
    )

    sim = method.make_simulation_full(mesh, survey)
    dmis = method.make_dmis(survey, sim)
    dmesh = mesh.to_discretize()

    dh = min(dmesh.h[0].min(), dmesh.h[1].min(), dmesh.h[2].min())
    if alpha_s is None:
        alpha_s = dh**-2

    reference_model = np.zeros(dmesh.n_cells)

    reg = regularization.WeightedLeastSquares(
        dmesh,
        alpha_s=alpha_s,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        reference_model=reference_model,
        reference_model_in_smooth=False,
    )

    opt = optimization.InexactGaussNewton(
        maxIter=max_iter, maxIterLS=20, cg_maxiter=10, cg_rtol=1e-2,
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    collector = IterationCollector()
    sens_weight = directives.UpdateSensitivityWeights(every_iteration=False)
    update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)

    directive_list = [
        sens_weight,
        update_jacobi,
        collector,
        directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
        directives.BetaSchedule(coolingFactor=cooling_factor, coolingRate=1),
        directives.TargetMisfit(chifact=1.0),
    ]

    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    print(f"    alpha_s={alpha_s:.4f}, beta0_ratio={beta0_ratio}")
    m_rec = inv.run(m0)

    result = InversionResult(method=method.method_name, converged=True)
    for snap in collector.snapshots:
        result.add_iteration(snap)
    result.final_model = PhysicalModel(mesh, m_rec, PhysicalProperty.DENSITY, "recovered")

    dpred = sim.dpred(m_rec)
    return result, m_rec, dpred


def main():
    np.random.seed(42)
    print("=" * 70)
    print("Sparse (IRLS) Gravity + Magnetics Inversion")
    print("  Padded mesh + SimPEG user-tutorial approach")
    print("=" * 70)

    # ── 1. Padded mesh ──────────────────────────────────────
    print("\n[1] Creating padded mesh and true models...")
    mesh = create_padded_mesh()
    nx, ny, nz = mesh.shape
    print(f"    Total mesh: ({nx}, {ny}, {nz}), {mesh.n_cells} cells")
    print(f"    Core: ({NX_CORE}, {NY_CORE}, {NZ_CORE}), padding={N_PAD} cells (factor={PAD_FACTOR})")
    print(f"    Core domain: {NX_CORE*DX:.0f}×{NY_CORE*DY:.0f}×{NZ_CORE*DZ:.0f} m")

    true_density, true_suscept = create_anomaly(mesh)

    # ── 2. Survey (on core domain) ──────────────────────────
    print("\n[2] Creating survey stations...")
    n_side = 15
    pad = DX * 1.5
    xs = np.linspace(pad, NX_CORE * DX - pad, n_side)
    ys = np.linspace(pad, NY_CORE * DY - pad, n_side)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_side**2)])
    n_stations = len(locs)
    print(f"    {n_stations} stations on {n_side}×{n_side} grid")
    print(f"    Station extent: x=[{xs.min():.0f},{xs.max():.0f}], "
          f"y=[{ys.min():.0f},{ys.max():.0f}] m")

    # ── 3. Forward modeling ─────────────────────────────────
    print("\n[3] Forward modeling...")
    grav_method = GravityMethod()
    mag_method = MagneticsMethod(inducing_field=(50000.0, 70.0, 0.0))

    true_density_model = PhysicalModel(mesh, true_density, PhysicalProperty.DENSITY, "true")
    true_suscept_model = PhysicalModel(mesh, true_suscept, PhysicalProperty.SUSCEPTIBILITY, "true")

    grav_survey_dummy = SurveyData(locs, np.zeros(n_stations), np.ones(n_stations), method="gravity")
    mag_survey_dummy = SurveyData(locs, np.zeros(n_stations), np.ones(n_stations), method="magnetics")

    grav_true = grav_method.forward(true_density_model, grav_survey_dummy)
    mag_true = mag_method.forward(true_suscept_model, mag_survey_dummy)

    print(f"    Gravity: [{grav_true.min():.4f}, {grav_true.max():.4f}] mGal")
    print(f"    Magnetics: [{mag_true.min():.2f}, {mag_true.max():.2f}] nT")

    # ── 4. Add noise ────────────────────────────────────────
    print("\n[4] Adding noise (2% + floor)...")
    grav_std = np.abs(grav_true) * 0.02 + 0.005
    grav_dobs = grav_true + grav_std * np.random.randn(n_stations)

    mag_std = np.abs(mag_true) * 0.02 + 0.5
    mag_dobs = mag_true + mag_std * np.random.randn(n_stations)

    grav_survey = SurveyData(locs, grav_dobs, grav_std, method="gravity")
    mag_survey = SurveyData(locs, mag_dobs, mag_std, method="magnetics")
    print(f"    Target phi_d = {n_stations} per method")

    # ── 5. L2 gravity (baseline) ────────────────────────────
    print("\n[5] L2 gravity inversion (baseline)...")
    m0_l2 = 1e-6 * np.ones(mesh.n_cells)
    l2_result, l2_rec, l2_pred = run_l2_inversion(
        grav_method, mesh, grav_survey, m0_l2,
        max_iter=100, beta0_ratio=10.0, cooling_factor=2.0,
    )
    print(f"    Iterations: {l2_result.n_iterations}")
    print(f"    phi_d: {l2_result.phi_d_history[-1]:.2f} (target {n_stations})")
    print(f"    Density: [{l2_rec.min():.4f}, {l2_rec.max():.4f}] (true max={BLOCK_DENSITY})")

    # ── 6. Sparse gravity (IRLS) ───────────────────────────
    print("\n[6] Sparse (L0+L1) gravity inversion...")
    m0_sparse = 1e-6 * np.ones(mesh.n_cells)
    sparse_result, sparse_rec, sparse_pred = run_sparse_inversion(
        grav_method, mesh, grav_survey, m0_sparse,
        max_irls_iterations=40,
        beta0_ratio=10.0,
        norms=[0, 1, 1, 1],
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    print(f"    Iterations: {sparse_result.n_iterations}")
    print(f"    phi_d: {sparse_result.phi_d_history[-1]:.2f}")
    print(f"    Density: [{sparse_rec.min():.4f}, {sparse_rec.max():.4f}] (true max={BLOCK_DENSITY})")

    # ── 7. Sparse magnetics ─────────────────────────────────
    print("\n[7] Sparse (L0+L1) magnetics inversion...")
    m0_mag = 1e-6 * np.ones(mesh.n_cells)
    mag_result, mag_rec, mag_pred = run_sparse_inversion(
        mag_method, mesh, mag_survey, m0_mag,
        max_irls_iterations=40,
        beta0_ratio=10.0,
        norms=[0, 1, 1, 1],
        lower_bound=-0.1,
        upper_bound=0.1,
    )
    print(f"    Iterations: {mag_result.n_iterations}")
    print(f"    phi_d: {mag_result.phi_d_history[-1]:.2f}")
    print(f"    Suscept: [{mag_rec.min():.5f}, {mag_rec.max():.5f}] (true max={BLOCK_SUSCEPT})")

    # ── 8. Summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    fmt = "{:22s} {:>12s} {:>12s} {:>12s}"
    print(fmt.format("", "True", "L2", "Sparse IRLS"))
    print("-" * 60)
    print(fmt.format("Density max",
        f"{true_density.max():.4f}",
        f"{l2_rec.max():.4f}",
        f"{sparse_rec.max():.4f}",
    ))
    print(fmt.format("Density min",
        f"{true_density.min():.4f}",
        f"{l2_rec.min():.4f}",
        f"{sparse_rec.min():.4f}",
    ))
    print(fmt.format("Grav phi_d",
        f"(target {n_stations})",
        f"{l2_result.phi_d_history[-1]:.1f}",
        f"{sparse_result.phi_d_history[-1]:.1f}",
    ))
    print(fmt.format("Suscept max",
        f"{true_suscept.max():.5f}",
        "-",
        f"{mag_rec.max():.5f}",
    ))
    print(fmt.format("Mag phi_d",
        f"(target {n_stations})",
        "-",
        f"{mag_result.phi_d_history[-1]:.1f}",
    ))

    # ── 9. 3D Viewers ───────────────────────────────────────
    print("\n[9] Generating 3D viewers...")

    grav_surface_l2 = {
        "method": "Gravity L2",
        "unit": "mGal",
        "obs_x": locs[:, 0].tolist(),
        "obs_y": locs[:, 1].tolist(),
        "obs_values": grav_dobs.tolist(),
        "pred_values": l2_pred.tolist(),
    }
    grav_surface_sparse = {
        "method": "Gravity Sparse",
        "unit": "mGal",
        "obs_x": locs[:, 0].tolist(),
        "obs_y": locs[:, 1].tolist(),
        "obs_values": grav_dobs.tolist(),
        "pred_values": sparse_pred.tolist(),
    }

    conv = {
        "Gravity L2": {
            "phi_d": l2_result.phi_d_history,
            "phi_m": l2_result.phi_m_history,
        },
        "Gravity Sparse": {
            "phi_d": sparse_result.phi_d_history,
            "phi_m": sparse_result.phi_m_history,
        },
    }

    out = generate_model3d_viewer(
        mesh=mesh,
        true_model=true_density,
        recovered_model=l2_rec,
        true_label="True Density",
        recovered_label="L2 Recovered",
        property_name="density",
        property_unit="g/cm³",
        surface_data=[grav_surface_l2],
        convergence={"Gravity L2": conv["Gravity L2"]},
        title="GeoInv3D — Gravity L2 Inversion",
        output_path="gravity_L2_viewer.html",
    )
    print(f"    {out}")

    out = generate_model3d_viewer(
        mesh=mesh,
        true_model=true_density,
        recovered_model=sparse_rec,
        true_label="True Density",
        recovered_label="Sparse (IRLS) Recovered",
        property_name="density",
        property_unit="g/cm³",
        surface_data=[grav_surface_sparse],
        convergence={"Gravity Sparse": conv["Gravity Sparse"]},
        title="GeoInv3D — Gravity Sparse IRLS Inversion",
        output_path="gravity_sparse_viewer.html",
    )
    print(f"    {out}")

    mag_surface = {
        "method": "Magnetics Sparse",
        "unit": "nT",
        "obs_x": locs[:, 0].tolist(),
        "obs_y": locs[:, 1].tolist(),
        "obs_values": mag_dobs.tolist(),
        "pred_values": mag_pred.tolist(),
    }
    out = generate_model3d_viewer(
        mesh=mesh,
        true_model=true_suscept,
        recovered_model=mag_rec,
        true_label="True Susceptibility",
        recovered_label="Sparse (IRLS) Recovered",
        property_name="susceptibility",
        property_unit="SI",
        surface_data=[mag_surface],
        convergence={"Magnetics Sparse": {
            "phi_d": mag_result.phi_d_history,
            "phi_m": mag_result.phi_m_history,
        }},
        title="GeoInv3D — Magnetics Sparse IRLS Inversion",
        output_path="magnetics_sparse_viewer.html",
    )
    print(f"    {out}")

    conv["Magnetics Sparse"] = {
        "phi_d": mag_result.phi_d_history,
        "phi_m": mag_result.phi_m_history,
    }
    out = generate_model3d_viewer(
        mesh=mesh,
        true_model=true_density,
        recovered_model=sparse_rec,
        true_label="True Density",
        recovered_label="Sparse Density",
        property_name="density",
        property_unit="g/cm³",
        surface_data=[grav_surface_l2, grav_surface_sparse, mag_surface],
        convergence=conv,
        title="GeoInv3D — All Methods Comparison",
        output_path="all_comparison_viewer.html",
    )
    print(f"    {out}")

    print("\nDone! Open the HTML files in a browser.")


if __name__ == "__main__":
    main()
