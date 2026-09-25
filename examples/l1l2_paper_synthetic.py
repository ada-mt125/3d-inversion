"""L1–L2 (elastic net) regularization on a magnetic synthetic, after Nwosu & Becken (2025).

Reproduces the set-up of the synthetic tests in Nwosu & Becken (2025, GJI 243,
ggaf390) on a coarser mesh and compares, through the same worker entry point the
cloud jobs use (``run_single_inversion``):

  * L1–L2 (``regularization_type="l1l2"``) for a = 0.1, 0.3, 0.5, 0.7
  * lp-norm sparse IRLS (``"sparse"``, the worker's default norms)
  * smooth L2 (``"l2"``)

each at noise levels sigma = 2, 5 and 10 nT.

Survey (as in the paper): TMI, B0 = 42000 nT, I = 45 deg, D = 30 deg,
26 x 26 stations at 2 km above flat ground, Gaussian noise sigma = 2 nT.
Model: one prism, chi = 0.03 SI.  The prism geometry and the mesh are ours
(the paper's exact geometry was not available here): 10 x 10 km, 2-10 km deep,
under the centre of a 50 x 50 km survey, on 2 x 2 x 1 km cells.

For each run it reports the recovered peak chi, compactness (volume above
10 % of the peak and share of the recovered chi·V inside the true prism), the
depth range of the anomaly (cells above half the peak; the 10 % range is in
the JSON) and the data misfit, then draws the figures and
writes ``results.json`` / ``results.md`` to the output directory.

Run with:
    python examples/l1l2_paper_synthetic.py [--out DIR] [--quick]

``--quick`` runs only sigma = 2 nT (about a third of the time).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geoinv3d.cloud.task import InversionTask
from geoinv3d.cloud.worker import run_single_inversion
from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.magnetics import MagneticsMethod

# --- Survey and model -----------------------------------------------------------
B0, INC, DEC = 42000.0, 45.0, 30.0
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

ALPHAS = (0.1, 0.3, 0.5, 0.7)
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


# --- Runs -----------------------------------------------------------------------
def method_label(reg_type: str, alpha: float | None) -> str:
    if reg_type == "l1l2":
        return f"L1–L2 a={alpha:g}"
    return {"sparse": "Sparse lp", "l2": "Smooth L2"}[reg_type]


def make_task(reg_type, locs, dobs, sigma, alpha=None) -> InversionTask:
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
    if reg_type in ("l1l2", "sparse"):
        # Susceptibility is non-negative; the smooth path has no bounds.
        task.bounds_lower, task.bounds_upper = 0.0, 1.0
    return task


def metrics(m, m_true, cc, vol, G, dobs, sigma) -> dict:
    peak = float(m.max())
    anomaly = m >= THRESHOLD * peak
    half = m >= 0.5 * peak
    depth = -cc[:, 2]
    inside = m_true > 0
    moment = np.clip(m, 0, None) * vol
    resid = (G @ m - dobs) / sigma
    return {
        "peak_chi": peak,
        "anomaly_volume_km3": float(vol[anomaly].sum() / 1e9),
        "moment_in_body": float(moment[inside].sum() / moment.sum()),
        # depth range of the half-peak anomaly, and of the 10 % anomaly
        "top_km": float((depth[half] - DZ / 2).min() / 1e3),
        "bottom_km": float((depth[half] + DZ / 2).max() / 1e3),
        "top10_km": float((depth[anomaly] - DZ / 2).min() / 1e3),
        "bottom10_km": float((depth[anomaly] + DZ / 2).max() / 1e3),
        "phi_d": float(resid @ resid),
        "chi2_per_datum": float(resid @ resid / len(dobs)),
        "rms_nT": float(np.sqrt(np.mean((G @ m - dobs) ** 2))),
        "model_error": float(np.linalg.norm(m - m_true) / np.linalg.norm(m_true)),
    }


def run_all(noise_levels, out_dir) -> dict:
    mesh = build_mesh()
    dmesh = mesh.to_discretize()
    cc, vol = dmesh.cell_centers, dmesh.cell_volumes
    m_true = true_model(cc)
    locs = build_stations()

    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = MagneticsMethod(inducing_field=(B0, INC, DEC)).make_simulation_full(mesh, survey)
    G = np.asarray(sim.G, dtype=float)
    d_clean = G @ m_true

    true_depth = {"top_km": BODY_DEPTH[0] / 1e3, "bottom_km": BODY_DEPTH[1] / 1e3,
                  "volume_km3": float(vol[m_true > 0].sum() / 1e9)}
    print(f"mesh {NX}x{NY}x{NZ} = {dmesh.n_cells} cells, {len(locs)} stations, "
          f"TMI range {d_clean.min():.1f} .. {d_clean.max():.1f} nT")

    configs = [("l1l2", a) for a in ALPHAS] + [("sparse", None), ("l2", None)]
    runs, models, data = [], {}, {}
    rng = np.random.default_rng(SEED)
    noise_unit = rng.standard_normal(len(locs))  # same noise pattern, scaled per level
    for sigma in noise_levels:
        dobs = d_clean + sigma * noise_unit
        data[sigma] = dobs
        for reg_type, alpha in configs:
            label = method_label(reg_type, alpha)
            t0 = time.time()
            result = run_single_inversion(make_task(reg_type, locs, dobs, sigma, alpha), mesh)
            m = np.asarray(result["recovered_model"], dtype=float)
            row = {"method": label, "reg_type": reg_type, "alpha": alpha, "sigma": sigma,
                   "n_iterations": result["n_iterations"], "runtime_s": time.time() - t0}
            row.update(metrics(m, m_true, cc, vol, G, dobs, sigma))
            runs.append(row)
            models[(label, sigma)] = m
            print(f"sigma={sigma:>4g}  {label:<14} peak={row['peak_chi']:.4f}  "
                  f"V10={row['anomaly_volume_km3']:7.0f} km3  "
                  f"in-body={row['moment_in_body']:.2f}  "
                  f"depth {row['top_km']:.1f}-{row['bottom_km']:.1f} km  "
                  f"chi2/N={row['chi2_per_datum']:.2f}  ({row['runtime_s']:.0f} s)")

    np.savez_compressed(
        os.path.join(out_dir, "models.npz"), m_true=m_true,
        **{f"{lab}|{s:g}": m for (lab, s), m in models.items()},
    )
    return {"runs": runs, "models": models, "data": data, "m_true": m_true,
            "d_clean": d_clean, "locs": locs, "dmesh": dmesh, "true": true_depth}


# --- Figures --------------------------------------------------------------------
# Categorical slots 1-6 of the reference palette, fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


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


def plot_sections(res, labels, sigmas, path, title):
    import matplotlib.pyplot as plt

    dmesh = res["dmesh"]
    xe, ze = dmesh.nodes_x / 1e3, dmesh.nodes_z / 1e3
    panels = [("True model", None)] + [(lab, s) for s in sigmas for lab in labels]
    ncol = len(labels) + 1 if len(sigmas) == 1 else len(labels)
    nrow = 1 if len(sigmas) == 1 else len(sigmas)
    if len(sigmas) > 1:
        panels = [(lab, s) for s in sigmas for lab in labels]
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    cmap = _chi_cmap()
    for ax, (lab, s) in zip(axes.ravel(), panels):
        m = res["m_true"] if s is None else res["models"][(lab, s)]
        vmax = max(float(m.max()), 1e-6)
        im = ax.pcolormesh(xe, ze, _section(m, dmesh), cmap=cmap, vmin=0, vmax=vmax)
        _draw_body(ax)
        name = lab if s is None or len(sigmas) == 1 else f"{lab}\nσ={s:g} nT"
        ax.set_title(f"{name}\npeak {vmax:.3f} SI", fontsize=8)
        ax.set_xlim(0, 50)
        ax.set_ylim(-20, 0)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.25, fraction=0.06)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([0, vmax])
        cb.ax.set_xticklabels(["0", f"{vmax:.3f}"])
        ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("z (km)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (km)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data(res, sigma, path):
    import matplotlib.pyplot as plt

    locs = res["locs"]
    shape = (N_STATIONS, N_STATIONS)
    x, y = locs[:, 0].reshape(shape) / 1e3, locs[:, 1].reshape(shape) / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, d, name in zip(axes, [res["d_clean"], res["data"][sigma]],
                           ["Noise-free TMI", f"Observed TMI (σ = {sigma:g} nT)"]):
        lim = float(abs(res["d_clean"]).max())
        im = ax.pcolormesh(x, y, d.reshape(shape), cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="nearest")
        ax.plot([BODY_X[0] / 1e3, BODY_X[1] / 1e3, BODY_X[1] / 1e3, BODY_X[0] / 1e3,
                 BODY_X[0] / 1e3],
                [BODY_Y[0] / 1e3, BODY_Y[0] / 1e3, BODY_Y[1] / 1e3, BODY_Y[1] / 1e3,
                 BODY_Y[0] / 1e3], "k--", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        fig.colorbar(im, ax=ax, label="nT", shrink=0.85)
    axes[0].set_ylabel("y (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics(res, labels, path):
    import matplotlib.pyplot as plt

    runs = res["runs"]
    sigmas = sorted({r["sigma"] for r in runs})
    panels = [("peak_chi", "Peak χ (SI)", CHI_TRUE),
              ("anomaly_volume_km3", "Volume ≥ 10 % of peak (km³)", res["true"]["volume_km3"]),
              ("moment_in_body", "Share of χ·V inside prism", 1.0),
              ("bottom_km", "Half-peak bottom depth (km)", res["true"]["bottom_km"]),
              ("chi2_per_datum", "χ² / N", 1.0)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (key, name, ref) in zip(axes, panels):
        for i, lab in enumerate(labels):
            vals = [next(r[key] for r in runs if r["method"] == lab and r["sigma"] == s)
                    for s in sigmas]
            ax.plot(sigmas, vals, color=SERIES_COLORS[i], marker=MARKERS[i], ms=6, lw=2,
                    label=lab)
        ax.axhline(ref, color="#52514e", lw=1, ls=":")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("σ (nT)", fontsize=8)
        ax.set_xticks(sigmas)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_yscale("log")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False)
    fig.text(0.995, 0.01, "dotted: true value", ha="right", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Report ---------------------------------------------------------------------
def write_tables(res, out_dir):
    runs = res["runs"]
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"true": res["true"], "chi_true": CHI_TRUE, "runs": runs}, f, indent=2)
    head = ("| σ (nT) | Method | Peak χ | V≥10% (km³) | χ·V in prism | Depth ≥50% (km) | "
            "χ²/N | RMS (nT) | ‖m−m*‖/‖m*‖ | Iter |")
    lines = [head, "|" + "---|" * 10]
    for r in runs:
        lines.append(
            f"| {r['sigma']:g} | {r['method']} | {r['peak_chi']:.4f} | "
            f"{r['anomaly_volume_km3']:.0f} | {r['moment_in_body']:.2f} | "
            f"{r['top_km']:.0f}–{r['bottom_km']:.0f} | {r['chi2_per_datum']:.2f} | "
            f"{r['rms_nT']:.2f} | {r['model_error']:.2f} | {r['n_iterations']} |")
    t = res["true"]
    lines.append(f"| – | True | {CHI_TRUE:.4f} | {t['volume_km3']:.0f} | 1.00 | "
                 f"{t['top_km']:.0f}–{t['bottom_km']:.0f} | 1 | σ | 0 | – |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output",
                                                      "l1l2_paper_synthetic"))
    parser.add_argument("--quick", action="store_true", help="sigma = 2 nT only")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    sigmas = NOISE_LEVELS[:1] if args.quick else NOISE_LEVELS
    res = run_all(sigmas, args.out)
    labels = [method_label("l1l2", a) for a in ALPHAS] + ["Sparse lp", "Smooth L2"]

    plot_data(res, sigmas[0], os.path.join(args.out, "data.png"))
    plot_sections(res, labels, sigmas[:1], os.path.join(args.out, "sections_alpha.png"),
                  f"E–W section at y = 25 km, σ = {sigmas[0]:g} nT (dashed: true prism)")
    if len(sigmas) > 1:
        plot_sections(res, labels, sigmas, os.path.join(args.out, "sections_noise.png"),
                      "E–W sections at y = 25 km by noise level (dashed: true prism)")
    plot_metrics(res, labels, os.path.join(args.out, "metrics.png"))
    print(write_tables(res, args.out))
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
