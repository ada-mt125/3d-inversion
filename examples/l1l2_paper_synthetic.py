"""L1–L2 regularization on a magnetic synthetic, following Utsugi (2019).

Method (Utsugi 2019, EPS 71:73): L1–L2 (elastic net) inversion solved by
coordinate descent along a decreasing lambda path (4 decades below lambda_max,
0.1 steps in log10, warm starts), lambda from the maximum curvature of the
L-curve, the mixing ratio alpha fixed a priori, and depth weighting wS1
(||k_j||^-1/2) or wS2 (1/||k_j||).  As in the paper, alpha is judged by the
distance to the true model, eps = ||b*_alpha - b*_true|| (magnetization, A/m),
over a range of alpha and noise levels.

Survey (Nwosu & Becken 2025, GJI 243 ggaf390): TMI, B0 = 42000 nT, I = 45 deg,
D = 30 deg, 26 x 26 stations at 2 km above flat ground, noise sigma = 2 nT
(also 5 and 10 nT).  Model: one prism, chi = 0.03 SI (1.00 A/m), 10 x 10 km,
2-10 km deep (our geometry), on 2 x 2 x 1 km cells.

Part A: eps(alpha) for both weightings and three noise levels (66 lambda paths).
Part B: L1–L2 at the best alpha of each weighting against the worker's lp-norm
sparse and smooth L2 inversions: peak chi, compactness, depth range, misfit.

Run with:
    python examples/l1l2_paper_synthetic.py [--out DIR] [--procs 4] [--quick]

``--quick`` runs sigma = 2 nT and five alphas only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# One BLAS thread per process: the alpha scan runs one process per core
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from geoinv3d.cloud.task import InversionTask  # noqa: E402
from geoinv3d.cloud.worker import run_single_inversion  # noqa: E402
from geoinv3d.datamodel.mesh import Mesh3D  # noqa: E402
from geoinv3d.datamodel.survey import SurveyData  # noqa: E402
from geoinv3d.methods.l1l2_cda import invert_l1l2  # noqa: E402
from geoinv3d.methods.magnetics import MagneticsMethod  # noqa: E402

# --- Survey and model -----------------------------------------------------------
B0, INC, DEC = 42000.0, 45.0, 30.0
H0 = B0 * 1e-9 / (4e-7 * np.pi)   # inducing field in A/m: magnetization = H0 * chi
CHI_TRUE = 0.03
STATION_HEIGHT = 2000.0
N_STATIONS = 26
SURVEY_EXTENT = 50_000.0  # m, 2 km station spacing

DX = DY = 2000.0
DZ = 1000.0
NX = NY = 28              # x, y from -2 to 54 km (one cell past the stations)
NZ = 20                   # 0 to 20 km depth
ORIGIN = (-2000.0, -2000.0, -NZ * DZ)

BODY_X = (20_000.0, 30_000.0)
BODY_Y = (20_000.0, 30_000.0)
BODY_DEPTH = (2000.0, 10_000.0)  # top, bottom (depth below ground)

ALPHAS = (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.99)
QUICK_ALPHAS = (0.3, 0.7, 0.9, 0.96, 0.99)
WEIGHTINGS = ("S1", "S2")
NOISE_LEVELS = (2.0, 5.0, 10.0)
THRESHOLD = 0.1           # fraction of the peak that counts as "anomaly"
SEED = 2025


def build_mesh() -> Mesh3D:
    return Mesh3D.uniform(NX, NY, NZ, DX, DY, DZ, origin=ORIGIN)


def build_stations() -> np.ndarray:
    xy = np.linspace(0.0, SURVEY_EXTENT, N_STATIONS)
    xx, yy = np.meshgrid(xy, xy)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, STATION_HEIGHT)])


def true_model(cc: np.ndarray) -> np.ndarray:
    depth = -cc[:, 2]
    inside = (
        (cc[:, 0] > BODY_X[0]) & (cc[:, 0] < BODY_X[1])
        & (cc[:, 1] > BODY_Y[0]) & (cc[:, 1] < BODY_Y[1])
        & (depth > BODY_DEPTH[0]) & (depth < BODY_DEPTH[1])
    )
    return inside * CHI_TRUE


def metrics(m, m_true, cc, vol, G, dobs, sigma) -> dict:
    peak = float(m.max())
    anomaly = m >= THRESHOLD * peak
    half = m >= 0.5 * peak
    depth = -cc[:, 2]
    inside = m_true > 0
    moment = np.clip(m, 0, None) * vol
    resid = G @ m - dobs
    return {
        "peak_chi": peak,
        "anomaly_volume_km3": float(vol[anomaly].sum() / 1e9),
        "moment_in_body": float(moment[inside].sum() / moment.sum()),
        # depth range of the half-peak anomaly, and of the 10 % anomaly
        "top_km": float((depth[half] - DZ / 2).min() / 1e3),
        "bottom_km": float((depth[half] + DZ / 2).max() / 1e3),
        "top10_km": float((depth[anomaly] - DZ / 2).min() / 1e3),
        "bottom10_km": float((depth[anomaly] + DZ / 2).max() / 1e3),
        "chi2_per_datum": float(np.sum((resid / sigma) ** 2) / len(dobs)),
        "residual_std_nT": float(resid.std()),
        "eps_Am": float(np.linalg.norm(H0 * (m - m_true))),   # Utsugi's model distance
        "model_error": float(np.linalg.norm(m - m_true) / np.linalg.norm(m_true)),
        "negative_share": float(-np.clip(m, None, 0).sum() / abs(m).sum()) + 0.0,
    }


# --- Worker tasks (also used by lambda_selection.py) -------------------------------
def method_label(reg_type: str, alpha: float | None) -> str:
    if reg_type == "l1l2":
        return f"L1–L2 a={alpha:g}"
    return {"sparse": "Sparse lp", "l2": "Smooth L2"}[reg_type]


def make_task(reg_type, locs, dobs, sigma, alpha=None, l1l2_solver="irls") -> InversionTask:
    """A worker task on this mesh; L1–L2 defaults to the IRLS solver here."""
    task = InversionTask(
        task_id=f"{reg_type}_{alpha}_{sigma}",
        nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, origin=ORIGIN,
        method_type="magnetics",
        method_kwargs={"inducing_field": (B0, INC, DEC), "component": "tmi"},
        regularization_type=reg_type,
        max_iter=60,
        max_irls_iterations=60,
        station_locations=locs,
        observed_data=dobs,
        data_std=np.full(len(dobs), sigma),
        initial_model=np.zeros(NX * NY * NZ),
    )
    if reg_type == "l1l2":
        task.l1_ratio = alpha
        task.l1l2_solver = l1l2_solver
    if reg_type in ("l1l2", "sparse"):
        # Susceptibility is non-negative; the smooth path has no bounds.
        task.bounds_lower, task.bounds_upper = 0.0, 1.0
    return task


# --- Shared problem (built in each process) -----------------------------------------
_P: dict = {}


def setup_problem() -> dict:
    mesh = build_mesh()
    dmesh = mesh.to_discretize()
    locs = build_stations()
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = MagneticsMethod(inducing_field=(B0, INC, DEC)).make_simulation_full(mesh, survey)
    G = np.asarray(sim.G, dtype=float)
    m_true = true_model(dmesh.cell_centers)
    d_clean = G @ m_true
    noise_unit = np.random.default_rng(SEED).standard_normal(len(locs))
    _P.update(mesh=mesh, dmesh=dmesh, cc=dmesh.cell_centers, vol=dmesh.cell_volumes,
              locs=locs, G=G, m_true=m_true, d_clean=d_clean, noise_unit=noise_unit)
    return _P


def observed(sigma: float) -> np.ndarray:
    return _P["d_clean"] + sigma * _P["noise_unit"]


def run_cda(job):
    """One L1–L2 lambda path with the L-curve choice (runs in a worker process)."""
    weighting, alpha, sigma = job
    d = observed(sigma)
    t0 = time.time()
    res, _ = invert_l1l2(_P["G"], d, alpha, weighting=weighting, model_unit=H0,
                         std=np.full(len(d), sigma), criterion="lcurve")
    row = {"method": f"L1–L2 {weighting}", "weighting": weighting, "alpha": alpha,
           "sigma": sigma, "lambda": res.lambda_opt,
           "lambda_discrepancy": res.lambda_discrepancy,
           "nnz": int(np.count_nonzero(res.model)), "sweeps": int(res.path.sweeps.sum()),
           "runtime_s": time.time() - t0, "warnings": res.warnings}
    row.update(metrics(res.model, _P["m_true"], _P["cc"], _P["vol"], _P["G"], d, sigma))
    curve = {"lambdas": res.path.lambdas.tolist(),
             "residual_norm": res.path.residual_norm.tolist(),
             "penalty": res.path.penalty.tolist(), "lambda_lcurve": res.lambda_lcurve,
             "lambda_discrepancy": res.lambda_discrepancy}
    return row, res.model, curve


def run_worker_method(reg_type, sigma):
    """The worker's lp-norm sparse or smooth L2 inversion (discrepancy principle)."""
    d = observed(sigma)
    task = make_task(reg_type, _P["locs"], d, sigma)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_single_inversion(task, _P["mesh"])
    m = np.asarray(result["recovered_model"], dtype=float)
    row = {"method": method_label(reg_type, None), "weighting": None, "alpha": None,
           "sigma": sigma, "runtime_s": time.time() - t0}
    row.update(metrics(m, _P["m_true"], _P["cc"], _P["vol"], _P["G"], d, sigma))
    return row, m


# --- Figures --------------------------------------------------------------------
# Categorical slots of the reference palette, fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def _section(m, dmesh):
    """E-W vertical section through the prism centre (y = 25 km)."""
    j = int(np.argmin(abs(dmesh.cell_centers_y - np.mean(BODY_Y))))
    return m.reshape(dmesh.shape_cells, order="F")[:, j, :].T


def _chi_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("chi", ["#fcfcfb", "#9ec5f0", "#2a78d6", "#0d2f5c"])


def _draw_body(ax):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((BODY_X[0] / 1e3, -BODY_DEPTH[1] / 1e3),
                           (BODY_X[1] - BODY_X[0]) / 1e3,
                           (BODY_DEPTH[1] - BODY_DEPTH[0]) / 1e3,
                           fill=False, ec="#0b0b0b", lw=1.0, ls="--"))


def _style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_eps(rows, sigmas, path):
    """eps(alpha) per weighting, one line per noise level (Utsugi's Figs 7, 10)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, w in zip(axes, WEIGHTINGS):
        for i, s in enumerate(sigmas):
            rs = sorted((r for r in rows if r["weighting"] == w and r["sigma"] == s),
                        key=lambda r: r["alpha"])
            a = [r["alpha"] for r in rs]
            e = [r["eps_Am"] for r in rs]
            ax.plot(a, e, color=SERIES_COLORS[i], marker=MARKERS[i], ms=6, lw=2,
                    label=f"σ = {s:g} nT")
            k = int(np.argmin(e))
            ax.plot(a[k], e[k], "o", ms=15, mfc="none", mec=SERIES_COLORS[i], mew=1.5)
        ax.set_title(f"Weighting w{w}", fontsize=10)
        ax.set_xlabel("α (mixing ratio)")
        _style(ax)
    axes[0].set_ylabel("ε = ‖β̂_α − β_true‖ (A/m)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Distance to the true model vs α (λ from the L-curve; rings: minima)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sections(panels, dmesh, path, title, ncol):
    """panels: list of (title, model)."""
    import matplotlib.pyplot as plt

    xe, ze = dmesh.nodes_x / 1e3, dmesh.nodes_z / 1e3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.35 * nrow + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    cmap = _chi_cmap()
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (name, m) in zip(axes.ravel(), panels):
        vmax = max(float(m.max()), 1e-6)
        im = ax.pcolormesh(xe, ze, _section(m, dmesh), cmap=cmap, vmin=0, vmax=vmax)
        _draw_body(ax)
        ax.set_title(f"{name}\npeak {vmax:.3f} SI", fontsize=8)
        ax.set_xlim(0, 50)
        ax.set_ylim(-20, 0)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("z (km)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (km)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lcurve(curve, row, path):
    """The L-curve and its curvature for one run (Utsugi's Fig. 4)."""
    import matplotlib.pyplot as plt
    from geoinv3d.methods.regparam import lcurve_curvature

    lam = np.asarray(curve["lambdas"])
    rn, pen = np.asarray(curve["residual_norm"]), np.asarray(curve["penalty"])
    ok = pen > 1e-6 * pen.max()
    tt, kappa = lcurve_curvature(lam[ok], rn[ok], pen[ok])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].loglog(rn[ok], pen[ok], color="#52514e", lw=1.5, marker="o", ms=3, mfc="none")
    for key, color, name in (("lambda_lcurve", "#eb6834", "L-curve corner"),
                             ("lambda_discrepancy", "#2a78d6", "χ² = N")):
        if curve[key] is None:
            continue
        i = int(np.argmin(abs(np.log(lam / curve[key]))))
        axes[0].plot(rn[i], pen[i], "o", color=color, ms=9, mec="#fcfcfb", label=name)
        axes[1].axvline(curve[key], color=color, lw=1.5, label=name)
    axes[0].set_xlabel("‖f − Xβ‖ (nT)")
    axes[0].set_ylabel("P(β; α)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].semilogx(np.exp(tt), kappa, color="#52514e", lw=1.5)
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("curvature")
    for ax in axes:
        _style(ax)
    fig.suptitle(f"L-curve, w{row['weighting']}, α = {row['alpha']:g}, "
                 f"σ = {row['sigma']:g} nT: λ̂ = {row['lambda']:.3g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics(rows, labels, path):
    import matplotlib.pyplot as plt

    sigmas = sorted({r["sigma"] for r in rows})
    vol_true = float(_P["vol"][_P["m_true"] > 0].sum() / 1e9)
    panels = [("peak_chi", "Peak χ (SI)", CHI_TRUE),
              ("anomaly_volume_km3", "Volume ≥ 10 % of peak (km³)", vol_true),
              ("moment_in_body", "Share of χ·V inside prism", 1.0),
              ("bottom_km", "Half-peak bottom depth (km)", BODY_DEPTH[1] / 1e3),
              ("eps_Am", "ε (A/m)", 0.0)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
    for ax, (key, name, ref) in zip(axes, panels):
        for i, lab in enumerate(labels):
            vals = [next(r[key] for r in rows if r["label"] == lab and r["sigma"] == s)
                    for s in sigmas]
            ax.plot(sigmas, vals, color=SERIES_COLORS[i], marker=MARKERS[i], ms=6, lw=2,
                    label=lab)
        ax.axhline(ref, color="#52514e", lw=1, ls=":")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("σ (nT)", fontsize=8)
        ax.set_xticks(sigmas)
        _style(ax)
    axes[1].set_yscale("log")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.text(0.995, 0.01, "dotted: true value", ha="right", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Tables ---------------------------------------------------------------------
def _row_md(r, name):
    return (f"| {r['sigma']:g} | {name} | {r['peak_chi']:.4f} | "
            f"{r['anomaly_volume_km3']:.0f} | {r['moment_in_body']:.2f} | "
            f"{r['top_km']:.0f}–{r['bottom_km']:.0f} | {r['chi2_per_datum']:.2f} | "
            f"{r['eps_Am']:.2f} | {r['negative_share']:.2f} |")


def write_tables(scan, compare, best, out_dir):
    head = ("| σ (nT) | Method | Peak χ | V≥10% (km³) | χ·V in prism | Depth ≥50% (km) | "
            "χ²/N | ε (A/m) | neg. share |")
    lines = ["## Part A: α scan (λ from the L-curve)", "", head, "|" + "---|" * 9]
    for r in sorted(scan, key=lambda r: (r["weighting"], r["sigma"], r["alpha"])):
        lines.append(_row_md(r, f"L1–L2 w{r['weighting']} α={r['alpha']:g} "
                                f"(λ̂={r['lambda']:.3g})"))
    lines += ["", "Best α (minimum ε summed over σ): "
              + ", ".join(f"w{w}: {a:g}" for w, a in best.items()), "",
              "## Part B: comparison", "", head, "|" + "---|" * 9]
    for r in compare:
        lines.append(_row_md(r, r["label"]))
    vol_true = float(_P["vol"][_P["m_true"] > 0].sum() / 1e9)
    lines.append(f"| – | True | {CHI_TRUE:.4f} | {vol_true:.0f} | 1.00 | "
                 f"{BODY_DEPTH[0] / 1e3:.0f}–{BODY_DEPTH[1] / 1e3:.0f} | 1 | 0 | 0 |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output",
                                                      "l1l2_paper_synthetic"))
    parser.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="sigma = 2 nT, five alphas")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    setup_problem()
    sigmas = NOISE_LEVELS[:1] if args.quick else NOISE_LEVELS
    alphas = QUICK_ALPHAS if args.quick else ALPHAS
    print(f"mesh {NX}x{NY}x{NZ} = {_P['dmesh'].n_cells} cells, {len(_P['locs'])} stations, "
          f"TMI {_P['d_clean'].min():.1f} .. {_P['d_clean'].max():.1f} nT, "
          f"true magnetization {CHI_TRUE * H0:.2f} A/m", flush=True)

    # Part A: alpha scan, longest jobs (large alpha) first
    jobs = [(w, a, s) for w in WEIGHTINGS for s in sigmas for a in alphas]
    jobs.sort(key=lambda j: (-j[1], j[0]))
    scan, models, curves = [], {}, {}
    t0 = time.time()
    # forkserver, not fork: thread pools started while computing G (BLAS, numba)
    # are not fork-safe, and forked workers can spin without progress.
    with get_context("forkserver").Pool(args.procs, initializer=setup_problem) as pool:
        for row, m, curve in pool.imap_unordered(run_cda, jobs):
            scan.append(row)
            key = (row["weighting"], row["alpha"], row["sigma"])
            models[key], curves[key] = m, curve
            warn = f" {row['warnings']}" if row["warnings"] else ""
            print(f"[{len(scan):>3}/{len(jobs)}] w{row['weighting']} α={row['alpha']:<5g} "
                  f"σ={row['sigma']:<4g} λ̂={row['lambda']:<8.3g} ε={row['eps_Am']:6.2f} "
                  f"peak={row['peak_chi']:.4f} χ²/N={row['chi2_per_datum']:.2f} "
                  f"({row['runtime_s']:.0f} s){warn}", flush=True)
    print(f"alpha scan: {time.time() - t0:.0f} s", flush=True)

    best = {}
    for w in WEIGHTINGS:
        total = {a: sum(r["eps_Am"] for r in scan if r["weighting"] == w and r["alpha"] == a)
                 for a in alphas}
        best[w] = min(total, key=total.get)

    # Part B: best alphas vs sparse and smooth L2
    compare, compare_models = [], {}
    for s in sigmas:
        for w in WEIGHTINGS:
            r = dict(next(r for r in scan if r["weighting"] == w and r["alpha"] == best[w]
                          and r["sigma"] == s))
            r["label"] = f"L1–L2 w{w} α={best[w]:g}"
            compare.append(r)
            compare_models[(r["label"], s)] = models[(w, best[w], s)]
        for reg_type in ("sparse", "l2"):
            r, m = run_worker_method(reg_type, s)
            r["label"] = r["method"]
            compare.append(r)
            compare_models[(r["label"], s)] = m
            print(f"σ={s:g} {r['label']}: peak={r['peak_chi']:.4f} ε={r['eps_Am']:.2f} "
                  f"χ²/N={r['chi2_per_datum']:.2f}", flush=True)
    labels = [f"L1–L2 w{w} α={best[w]:g}" for w in WEIGHTINGS] + ["Sparse lp", "Smooth L2"]

    # Outputs
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"best_alpha": best, "scan": scan, "compare": compare,
                   "curves": {f"{k[0]}|{k[1]:g}|{k[2]:g}": v for k, v in curves.items()}},
                  f, indent=2)
    np.savez_compressed(os.path.join(args.out, "models.npz"), m_true=_P["m_true"],
                        **{f"{w}|{a:g}|{s:g}": m for (w, a, s), m in models.items()})

    dmesh = _P["dmesh"]
    plot_eps(scan, sigmas, os.path.join(args.out, "eps_vs_alpha.png"))
    s0 = sigmas[0]
    for w in WEIGHTINGS:
        panels = [("True model", _P["m_true"])] + [
            (f"w{w} α={a:g}", models[(w, a, s0)]) for a in alphas]
        plot_sections(panels, dmesh, os.path.join(args.out, f"sections_alpha_{w}.png"),
                      f"L1–L2 w{w}, σ = {s0:g} nT, λ from the L-curve: E–W section at "
                      "y = 25 km (dashed: true prism)", ncol=6)
    panels = [(f"{lab}\nσ={s:g} nT", compare_models[(lab, s)]) for s in sigmas
              for lab in labels]
    plot_sections(panels, dmesh, os.path.join(args.out, "sections_compare.png"),
                  "Best-α L1–L2 vs sparse and smooth L2 (dashed: true prism)",
                  ncol=len(labels))
    w_best = min(WEIGHTINGS, key=lambda w: sum(
        r["eps_Am"] for r in compare if r["label"] == f"L1–L2 w{w} α={best[w]:g}"))
    key = (w_best, best[w_best], s0)
    plot_lcurve(curves[key], next(r for r in scan if (r["weighting"], r["alpha"],
                                                      r["sigma"]) == key),
                os.path.join(args.out, "lcurve.png"))
    plot_metrics(compare, labels, os.path.join(args.out, "metrics.png"))
    print(write_tables(scan, compare, best, args.out))
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
