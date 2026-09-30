"""Figures and numbers of the Karnataka magnetic report.

    py examples/output/karnataka_magnetic/scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from kmodel import AOI, Grid, box_stats, data_fit, gridded, load, prep  # noqa: E402

DATA, FIGS = ROOT / "data" / "ec2_runs", ROOT / "figures"
if not DATA.exists():
    DATA = ROOT / "data" / "ec2_trials"
GRAVITY = ROOT.parent / "karnataka_gravity_terrain" / "data" / "ec2_runs" / "as1_beta1"
FULL = [
    # key, label, colour, line style
    ("sens", "sparse, sensitivity weighting", "#7f7f7f", "-"),
    ("beta0.5", "sparse, β=0.5", "#2e9c6a", "--"),
    ("beta1", "sparse, β=1", "#2e9c6a", "-"),
    ("beta1.5", "sparse, β=1.5", "#2e9c6a", ":"),
    ("beta2", "sparse, β=2", "#8e44ad", "-"),
    ("beta3", "sparse, β=3", "#8e44ad", ":"),
    ("l1l2_irls", "L1–L2 (IRLS)", "#1f5f8b", "-"),
]
TESTS = [
    ("beta1", "β=1 (reference)"),
    ("as0.1_beta1", "α_s = 0.1"),
    ("beta1_ub0.3", "upper bound 0.3 SI"),
    ("beta1_err2", "error 2 % + 5 nT"),
    ("beta1_h500", "continued to 500 m"),
    ("beta1_raw80", "as flown, 80 m (not continued)"),
    ("beta1_flat", "flat earth"),
]
SLICE_RUNS = ["sens", "beta0.5", "beta1", "beta1.5", "l1l2_irls"]
SANDUR = (669500.0, 1667500.0)            # the strongest magnetic body (Sandur belt)
BOXES = {"sandur": (658000.0, 678000.0, 1657000.0, 1675000.0),
         "ne": (682000.0, 702000.0, 1660000.0, 1682000.0)}
def chi_style():
    """Susceptibility colours: a square-root scale, so that 0.05 SI and the 1 SI bound both show."""
    from matplotlib.colors import PowerNorm
    return dict(cmap="Oranges", norm=PowerNorm(0.5, vmin=0.0, vmax=1.0))


CHI_TICKS = [0, 0.05, 0.2, 0.5, 1.0]


def measures(g, d):
    return {**g.shares(), **data_fit(d), **{f"box_{k}": box_stats(g, b) for k, b in BOXES.items()},
            "model_max": float(g.m.max()), "n_active": int(g.active.sum()),
            "at_bound": float(np.mean(g.m[g.active] > 0.98 * g.m.max()))}


def imshow(ax, g, x, y, **kw):
    if y[0] > y[-1]:
        g, y = g[::-1], y[::-1]
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    return ax.imshow(g, origin="lower", extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy), **kw)


def boxes(ax):
    for b in BOXES.values():
        ax.plot(np.array([b[0], b[1], b[1], b[0], b[0]]) / 1e3, np.array([b[2], b[2], b[3], b[3], b[2]]) / 1e3,
                color="k", lw=0.7, ls="--")


def main():
    FIGS.mkdir(exist_ok=True)
    pj, pz = prep()
    keys = [k for k, *_ in FULL] + [k for k, _ in TESTS if k != "beta1"]
    runs = {k: load(DATA / k) for k in keys}
    grids = {k: Grid(r) for k, r in runs.items()}
    ref, gref = runs["beta1"], grids["beta1"]
    d = ref["_data"]
    numbers = {"prep": pj, "full": {}, "runs": {},
               "data": {"n": int(len(d["observed"])),
                        "raw": [float((d["observed"] + d["regional"]).min()), float((d["observed"] + d["regional"]).max())],
                        "regional": [float(d["regional"].min()), float(d["regional"].max())],
                        "residual": [float(d["observed"].min()), float(d["observed"].max())],
                        "std": [float(d["std"].min()), float(np.median(d["std"])), float(d["std"].max())]},
               "mesh": {"shape": list(gref.shape), "n_cells": int(gref.mesh.n_cells), "n_active": int(gref.active.sum()),
                        "ground_min": float(gref.ground[gref.core_xy].min()), "ground_max": float(gref.ground[gref.core_xy].max()),
                        "z_bottom": float(gref.mesh.nodes_z[0]), "z_core_base": gref.z_core_base},
               "receivers": [float(d["locations"][:, 2].min()), float(d["locations"][:, 2].max())]}
    for k, r in runs.items():
        numbers["full"][k] = {**measures(grids[k], r["_data"]), "settings": r["settings"],
                              "n_iterations": r.get("n_iterations"),
                              "noise": [r["datasets"][0]["noise_pct"], r["datasets"][0]["noise_floor"]]}
        run = DATA / k / "run.json"
        if run.exists():
            numbers["runs"][k] = json.loads(run.read_text())

    integ, cent = {}, {}
    for k in runs:
        integ[k], cent[k] = grids[k].integrated()
    fk = [k for k, *_ in FULL]
    C = np.corrcoef(np.array([integ[k].ravel() for k in fk]))
    numbers["integrated_corr"] = {"keys": fk, "matrix": np.round(C, 3).tolist()}
    # compact models put thin sheets a cell apart: compare the maps at 5 km resolution as well
    from scipy.ndimage import gaussian_filter
    smooth = {k: gaussian_filter(integ[k], 2.0) for k in runs}
    Cs = np.corrcoef(np.array([smooth[k].ravel() for k in fk]))
    numbers["integrated_corr_smooth"] = {"keys": fk, "matrix": np.round(Cs, 3).tolist()}
    numbers["integrated_tests_smooth"] = {k: float(np.corrcoef(smooth[k].ravel(), smooth["beta1"].ravel())[0, 1]) for k, _ in TESTS}
    # the bodies at the edges of the area (outside both boxes and within 8 km of the core's edge)
    cx, cy = gref.mesh.cell_centers_x[gref.sx], gref.mesh.cell_centers_y[gref.sy]
    edge = ((cx < AOI[0] + 8000) | (cx > AOI[1] - 8000))[:, None] | ((cy < AOI[2] + 8000) | (cy > AOI[3] - 8000))[None, :]
    numbers["edge_share"] = {k: float(integ[k][edge].sum() / integ[k].sum()) for k in runs}
    numbers["edge_area_share"] = float(edge.mean())
    numbers["integrated_tests"] = {k: float(np.corrcoef(integ[k].ravel(), integ["beta1"].ravel())[0, 1]) for k, _ in TESTS}
    # residual structure: where the misfit is, and how much of it
    res = (d["observed"] - d["predicted"]) / d["std"]
    numbers["residual"] = {"p01": float(np.percentile(res, 1)), "p99": float(np.percentile(res, 99)),
                           "worst_1pct_share": float(np.sort(res ** 2)[-len(res) // 100:].sum() / (res ** 2).sum()),
                           "above_3": float(np.mean(np.abs(res) > 3)),
                           # nodes (1 km2 each) where each model underfits by more than 150 nT
                           "n_above_150": {k: int(np.sum(runs[k]["_data"]["observed"] - runs[k]["_data"]["predicted"] > 150))
                                           for k, *_ in FULL},
                           "n_below_150": {k: int(np.sum(runs[k]["_data"]["observed"] - runs[k]["_data"]["predicted"] < -150))
                                           for k, *_ in FULL}}
    # against the gravity model (terrain, β = 1): maps of the same area
    gg = Grid(load(GRAVITY))
    gi, _ = gg.integrated()
    pos = np.clip(gi, 0, None)
    numbers["gravity"] = {
        "corr_integrated": float(np.corrcoef(gi.ravel(), integ["beta1"].ravel())[0, 1]),
        "corr_positive": float(np.corrcoef(pos.ravel(), integ["beta1"].ravel())[0, 1]),
        "corr_l1l2": float(np.corrcoef(pos.ravel(), integ["l1l2_irls"].ravel())[0, 1]),
        # of the magnetic columns (top fifth by integrated susceptibility), how many are dense too
        "magnetic_in_dense": float(np.mean(gi[integ["beta1"] > np.percentile(integ["beta1"], 80)] > 0.3)),
        "dense_in_magnetic": float(np.mean(integ["beta1"][gi > 0.3] > np.percentile(integ["beta1"], 80))),
        "box_sandur_density_centroid": box_stats(Grid(load(GRAVITY), np.clip(load(GRAVITY)["_model"], 0, None)), BOXES["sandur"])["centroid_km"]}

    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                         "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
    xe = gref.mesh.nodes_x[gref.n_pad:gref.shape[0] - gref.n_pad + 1] / 1e3
    ye = gref.mesh.nodes_y[gref.n_pad:gref.shape[1] - gref.n_pad + 1] / 1e3

    # ── the data: as flown, continued, inverted ──
    gx, gy, g_res = gridded(d["locations"], d["observed"])
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={"wspace": 0.3})
    im = imshow(axs[0], pz["dem"], pz["dem_x"] / 1e3, pz["dem_y"] / 1e3, cmap="terrain", vmin=300, vmax=1100)
    plt.colorbar(im, ax=axs[0], shrink=0.8, label="m"); axs[0].set_title("Ground elevation")
    im = imshow(axs[1], pz["tmi_raw"], pz["tmi_x"] / 1e3, pz["tmi_y"] / 1e3, cmap="RdBu_r", vmin=-1500, vmax=1500)
    plt.colorbar(im, ax=axs[1], shrink=0.8, label="nT", extend="both"); axs[1].set_title("TMI anomaly as flown (80 m above the ground)")
    im = imshow(axs[2], pz["tmi_1000"], pz["tmi_x"] / 1e3, pz["tmi_y"] / 1e3, cmap="RdBu_r", vmin=-1000, vmax=1000)
    plt.colorbar(im, ax=axs[2], shrink=0.8, label="nT", extend="both"); axs[2].set_title("Continued upwards to 1 km above the ground")
    im = imshow(axs[3], g_res, gx, gy, cmap="RdBu_r", vmin=-1000, vmax=1000)
    plt.colorbar(im, ax=axs[3], shrink=0.8, label="nT", extend="both"); axs[3].set_title("At the 1 km nodes, trend removed (inverted)")
    for ax in axs:
        ax.set_xlabel("Easting (km, UTM 43N)"); ax.set_aspect("equal")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "data.png"); plt.close(fig)

    # ── residual maps ──
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.6), layout="constrained")
    for ax, (k, lab, *_) in zip(axs.ravel(), FULL):
        dd = runs[k]["_data"]
        x, y, g = gridded(dd["locations"], dd["observed"] - dd["predicted"])
        im = imshow(ax, g, x, y, cmap="RdBu_r", vmin=-150, vmax=150)
        f = numbers["full"][k]
        ax.set_title(f"{lab}\nχ²/N {f['chi2']:.2f} · RMS {f['rms']:.0f} nT", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    x, y, g = gridded(d["locations"], d["std"])
    ax = axs.ravel()[7]
    im2 = imshow(ax, g, x, y, cmap="Greys", vmin=0, vmax=80)
    ax.set_title("Data error, 5 % + 10 nT", fontsize=9); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.colorbar(im2, ax=ax, shrink=0.75, label="nT", extend="max", location="right")
    fig.colorbar(im, ax=axs[:, :], shrink=0.4, label="Residual, observed − predicted (nT)", extend="both",
                 location="bottom", aspect=40)
    fig.savefig(FIGS / "residuals.png"); plt.close(fig)

    # ── depth slices ──
    fig, axs = plt.subplots(len(SLICE_RUNS), 3, figsize=(10, 3.0 * len(SLICE_RUNS)), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.08, "hspace": 0.25})
    for r, k in enumerate(SLICE_RUNS):
        lab = next(l for kk, l, *_ in FULL if kk == k)
        for c, depth_km in enumerate([1.0, 3.0, 6.0]):
            x, y, vv, dk = grids[k].depth_slice(depth_km)
            ax = axs[r, c]
            im = ax.pcolormesh(x, y, vv.T, **chi_style())
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
            ax.set_title((lab if c == 0 else "") + f"\n{dk:.1f} km below the mean ground", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=axs, shrink=0.4, label="Susceptibility (SI)", ticks=CHI_TICKS)
    fig.savefig(FIGS / "slices.png"); plt.close(fig)

    def draw_section(ax, g, title, northing=SANDUR[1]):
        x, z, v, ground, n = g.section(northing)
        im = ax.pcolormesh(x, z, v, **chi_style())
        ax.plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
        ax.axhline(g.z_core_base / 1e3, color="k", lw=0.7, ls="--")
        ax.set_ylim(z.min(), 1.6); ax.set_ylabel("Elevation (km)")
        ax.set_title(f"{title} — northing {n / 1e3:.1f} km", loc="left")
        return im

    # ── E-W sections through the Sandur belt ──
    fig, axs = plt.subplots(len(FULL), 1, figsize=(10, 1.9 * len(FULL)), sharex=True)
    for ax, (k, lab, *_) in zip(axs, FULL):
        im = draw_section(ax, grids[k], lab)
    axs[-1].set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im, ax=axs, shrink=0.5, label="Susceptibility (SI)", ticks=CHI_TICKS)
    fig.savefig(FIGS / "sections.png"); plt.close(fig)

    # ── integrated susceptibility and its centroid depth ──
    fig, axs = plt.subplots(2, len(SLICE_RUNS), figsize=(16, 6.6), sharex=True, sharey=True, layout="constrained")
    for c, k in enumerate(SLICE_RUNS):
        a = axs[0, c].pcolormesh(xe, ye, integ[k].T, cmap="Oranges", vmin=0, vmax=1.0)
        b = axs[1, c].pcolormesh(xe, ye, cent[k].T, cmap="viridis_r", vmin=0, vmax=15)
        axs[0, c].set_title(next(l for kk, l, *_ in FULL if kk == k), fontsize=8.5)
        for ax in axs[:, c]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        boxes(axs[0, c])
    fig.colorbar(a, ax=axs[0], shrink=0.8, label="Integrated susceptibility\n(SI·km)", extend="max")
    fig.colorbar(b, ax=axs[1], shrink=0.8, label="Centroid depth below\nthe ground (km)")
    fig.savefig(FIGS / "robust.png"); plt.close(fig)

    # ── moment with depth ──
    edges = np.arange(0, 17.01, 0.5)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, lab, colr, ls in FULL:
        ax.plot(grids[k].mass_profile(edges) * 100, mid, ls, color=colr, lw=1.8, label=lab)
    ax.axhline(10, color="k", lw=0.8, ls="--")
    ax.set_xlabel("share of susceptibility × volume (%/km, below the core area)")
    ax.set_ylabel("Depth below the ground (km)"); ax.set_ylim(17, 0)
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIGS / "mass_profiles.png"); plt.close(fig)

    # ── the tests at β = 1 ──
    fig, axs = plt.subplots(len(TESTS), 2, figsize=(12.5, 1.9 * len(TESTS)), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.12, "hspace": 0.55})
    for r, (k, lab) in enumerate(TESTS):
        f = numbers["full"][k]
        im = draw_section(axs[r, 0], grids[k], f"{lab} · χ²/N {f['chi2']:.2f}")
        a = axs[r, 1].pcolormesh(xe, ye, integ[k].T, cmap="Oranges", vmin=0, vmax=1.0)
        axs[r, 1].set_aspect("equal"); axs[r, 1].tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im, ax=axs[:, 0], shrink=0.4, label="Susceptibility (SI)", ticks=CHI_TICKS, location="left", pad=0.09)
    fig.colorbar(a, ax=axs[:, 1], shrink=0.4, label="Integrated susceptibility (SI·km)", extend="max")
    fig.savefig(FIGS / "tests.png"); plt.close(fig)

    # ── magnetics against gravity ──
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained")
    a = axs[0].pcolormesh(xe, ye, gi.T, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    fig.colorbar(a, ax=axs[0], shrink=0.8, label="g/cc·km"); axs[0].set_title("Integrated density contrast\n(gravity, β = 1)")
    b = axs[1].pcolormesh(xe, ye, integ["beta1"].T, cmap="Oranges", vmin=0, vmax=1.0)
    fig.colorbar(b, ax=axs[1], shrink=0.8, label="SI·km", extend="max"); axs[1].set_title("Integrated susceptibility\n(magnetics, β = 1)")
    c = axs[2].pcolormesh(xe, ye, gi.T, cmap="RdBu_r", vmin=-1.5, vmax=1.5, alpha=0.55)
    axs[2].contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, smooth["beta1"].T, levels=[0.15, 0.4], colors=["#b24a1c", "#5a1f06"], linewidths=[0.9, 1.3])
    fig.colorbar(c, ax=axs[2], shrink=0.8, label="g/cc·km")
    axs[2].set_title("Susceptibility (contours, smoothed over 5 km:\n0.15 and 0.4 SI·km) on the density")
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlabel("Easting (km, UTM 43N)")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "gravity.png"); plt.close(fig)

    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print("figures and numbers written to", FIGS)


if __name__ == "__main__":
    main()
