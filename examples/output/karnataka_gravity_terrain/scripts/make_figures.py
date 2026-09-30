"""Figures and numbers of the Karnataka gravity report with terrain.

    py examples/output/karnataka_gravity_terrain/scripts/make_figures.py
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
from kmodel import AOI, MAIN_HIGH, NW_HIGH, Grid, body, data_fit, gridded, load, prep  # noqa: E402

sys.path.insert(0, str(ROOT.parents[2]))
from geoinv3d.methods.regional import fit_trend  # noqa: E402

DATA, FIGS = ROOT / "data", ROOT / "figures"
OLD = ROOT.parent / "karnataka_gravity" / "data" / "ec2_runs"      # the flat-earth runs (28 Sep)
FULL = [
    # key, label, colour, line style
    ("original_sparse", "Original sparse (α_s=1e-4, sensitivity)", "#7f7f7f", "-"),
    ("l1l2_irls", "L1–L2 (IRLS)", "#1f5f8b", "-"),
    ("as1_beta0.5", "sparse α_s=1, β=0.5", "#2e9c6a", "--"),
    ("as1_beta1", "sparse α_s=1, β=1", "#2e9c6a", "-"),
    ("as1_beta1.5", "sparse α_s=1, β=1.5", "#2e9c6a", ":"),
    ("as0.1_beta1", "sparse α_s=0.1, β=1", "#d68910", "-"),
]
STEPS = [   # what each change does, at α_s = 1, β = 1
    ("flat500", "Flat earth, 500 m layers, no terrain correction (28 Sep report)"),
    ("as1_beta1_flat", "Flat earth, 250 m layers, no terrain correction"),
    ("as1_beta1_no_tc", "Terrain in the mesh, no terrain correction"),
    ("as1_beta1", "Terrain in the mesh and terrain correction (this report)"),
]
SLICE_RUNS = ["original_sparse", "l1l2_irls", "as1_beta0.5", "as1_beta1", "as1_beta1.5"]
LOWRES = ["base", "as1e-2", "as0.1", "as1", "as1_p0111", "as1_p0222", "as1_p0220", "as1e-4_p0222",
          "as1_p0222_dw0.5", "as1_p0222_dw1", "as1_p0222_dw1.5", "as1_p0222_dw2",
          "as1_p0221_dw1", "as1_p0111_dw1", "as0.1_p0222_dw1", "base_dw1"]
LOW_PROFILES = [("base", "α_s=1e-4, [0,2,2,1], sens", "#7f7f7f"), ("as1_p0222", "α_s=1, [0,2,2,2], sens", "#d68910"),
                ("as1_p0222_dw2", "α_s=1, β=2", "#8e44ad"), ("as1_p0222_dw1", "α_s=1, β=1", "#2e9c6a"),
                ("as1_p0222_dw0.5", "α_s=1, β=0.5", "#1f5f8b")]
EXT = (AOI[0] / 1e3, AOI[1] / 1e3, AOI[2] / 1e3, AOI[3] / 1e3)


def measures(g, d):
    z, v, dz = g.column(MAIN_HIGH)
    zn, vn, dzn = g.column(NW_HIGH)
    return {**g.shares(), **data_fit(d), "main": body(z, v, dz), "nw": body(zn, vn, dzn),
            "ground_main_m": float(g.ground[g.ij(MAIN_HIGH)]), "ground_nw_m": float(g.ground[g.ij(NW_HIGH)]),
            "n_cells": int(g.mesh.n_cells), "n_active": int(g.active.sum()), "shape": list(g.shape)}


def imshow(ax, g, x, y, **kw):
    if y[0] > y[-1]:                        # rows north to south
        g, y = g[::-1], y[::-1]
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    return ax.imshow(g, origin="lower", extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy), **kw)


def main():
    FIGS.mkdir(exist_ok=True)
    pj, pz = prep()
    runs = {k: load(DATA / "ec2_runs" / k) for k, *_ in FULL}
    runs["as1_beta1_no_tc"] = load(DATA / "ec2_runs" / "as1_beta1_no_tc")
    runs["as1_beta1_flat"] = load(DATA / "ec2_runs" / "as1_beta1_flat")
    old = {k: load(OLD / k) for k, *_ in FULL}
    low = {k: load(DATA / "lowres_runs" / k) for k in LOWRES if (DATA / "lowres_runs" / k / "result.zip").exists()}
    grids = {k: Grid(r) for k, r in runs.items()}
    ogrids = {k: Grid(r) for k, r in old.items()}
    lgrids = {k: Grid(r) for k, r in low.items()}
    ref = runs["as1_beta1"]
    d = ref["_data"]

    numbers = {"prep": pj, "full": {}, "old": {}, "low": {}, "steps": {},
               "data": {"n": int(len(d["observed"])),
                        "raw": [float((d["observed"] + d["regional"]).min()), float((d["observed"] + d["regional"]).max())],
                        "regional": [float(d["regional"].min()), float(d["regional"].max())],
                        "residual": [float(d["observed"].min()), float(d["observed"].max())]},
               "mesh": {"shape": list(grids["as1_beta1"].shape), "n_cells": int(grids["as1_beta1"].mesh.n_cells),
                        "n_active": int(grids["as1_beta1"].active.sum()),
                        "ground_min": float(grids["as1_beta1"].ground[grids["as1_beta1"].core_xy].min()),
                        "ground_max": float(grids["as1_beta1"].ground[grids["as1_beta1"].core_xy].max()),
                        "ground_mean": grids["as1_beta1"].ground_mean,
                        "z_bottom": float(grids["as1_beta1"].mesh.nodes_z[0]),
                        "z_core_base": grids["as1_beta1"].z_core_base,
                        "lifted": (ref["topography"].get("stations_lifted") or {}).get("gravity")},
               "runs": {}}
    for k, r in runs.items():
        numbers["full"][k] = {**measures(grids[k], r["_data"]), "settings": r["settings"],
                              "n_iterations": r.get("n_iterations")}
        run = DATA / "ec2_runs" / k / "run.json"
        if run.exists():
            numbers["runs"][k] = json.loads(run.read_text())
    for k, r in old.items():
        numbers["old"][k] = measures(ogrids[k], r["_data"])
    for k, r in low.items():
        numbers["low"][k] = {**measures(lgrids[k], r["_data"]), "settings": r["settings"]}
    step_grids = {"flat500": ogrids["as1_beta1"], **{k: grids[k] for k, _ in STEPS[1:]}}
    step_data = {"flat500": old["as1_beta1"]["_data"], **{k: runs[k]["_data"] for k, _ in STEPS[1:]}}
    for k, _ in STEPS:
        numbers["steps"][k] = measures(step_grids[k], step_data[k])

    integ, cent = {}, {}
    for k in runs:
        integ[k], cent[k] = grids[k].integrated()
    keys = [k for k, *_ in FULL]
    C = np.corrcoef(np.array([integ[k].ravel() for k in keys]))
    numbers["integrated_corr"] = {"keys": keys, "matrix": np.round(C, 3).tolist(),
                                  "min_offdiag": float(C[~np.eye(len(keys), dtype=bool)].min())}
    oi = {k: ogrids[k].integrated()[0] for k in keys}
    numbers["integrated_old_vs_new"] = {k: float(np.corrcoef(oi[k].ravel(), integ[k].ravel())[0, 1]) for k in keys}
    si = {k: step_grids[k].integrated()[0] for k, _ in STEPS}
    numbers["integrated_steps"] = {k: float(np.corrcoef(si[k].ravel(), si["as1_beta1"].ravel())[0, 1]) for k, _ in STEPS}
    # the top 1 km below the ground with and without the terrain correction, in the hills
    gt, gn = grids["as1_beta1"], grids["as1_beta1_no_tc"]
    tc = pz["tc_nodes"]
    xs, ys, g_tc = pz["node_x"] / 1e3, pz["node_y"] / 1e3, tc
    top = lambda g: np.nansum(np.where((g.depth < 1000) & g.active, g.m, 0.0) * g.dz[None, None, :], axis=2)[g.sx, g.sy] / 1e3  # noqa: E731
    shallow = {"tc": top(gt), "no_tc": top(gn)}
    numbers["shallow"] = {"diff_max": float((shallow["tc"] - shallow["no_tc"]).max()),
                          "diff_min": float((shallow["tc"] - shallow["no_tc"]).min())}

    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                         "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

    # ── terrain and its correction ──
    X, Y = np.meshgrid(pz["node_x"], pz["node_y"])
    ba = pz["ba"]
    ok = np.isfinite(ba)

    def residual(v):
        out = np.full(v.shape, np.nan)
        out[ok] = v[ok] - fit_trend(np.column_stack([X[ok], Y[ok]]), v[ok], 2)
        return out
    change = residual(ba + tc) - residual(ba)
    numbers["tc_change"] = {"min": float(np.nanmin(change)), "max": float(np.nanmax(change)),
                            "at_main_high": float(change[np.argmin(np.abs(pz["node_y"] - MAIN_HIGH[1])),
                                                         np.argmin(np.abs(pz["node_x"] - MAIN_HIGH[0]))])}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"wspace": 0.32})
    im = imshow(axs[0], pz["dem"], pz["dem_x"] / 1e3, pz["dem_y"] / 1e3, cmap="terrain", vmin=300, vmax=1100)
    plt.colorbar(im, ax=axs[0], shrink=0.8, label="m")
    inside = (pz["station_x"] >= AOI[0]) & (pz["station_x"] <= AOI[1]) & (pz["station_y"] >= AOI[2]) & (pz["station_y"] <= AOI[3])
    axs[0].plot(pz["station_x"][inside] / 1e3, pz["station_y"][inside] / 1e3, ".", ms=0.9, color="k", alpha=0.55)
    axs[0].set_title("Ground elevation (Copernicus GLO-90) and the gravity stations")
    im = imshow(axs[1], g_tc, xs, ys, cmap="magma_r", vmin=0, vmax=5)
    plt.colorbar(im, ax=axs[1], shrink=0.8, label="mGal", extend="max")
    axs[1].set_title("Terrain correction at the 1 km nodes")
    im = imshow(axs[2], change, xs, ys, cmap="RdBu_r", vmin=-3, vmax=3)
    plt.colorbar(im, ax=axs[2], shrink=0.8, label="mGal", extend="max")
    axs[2].set_title("Change of the inverted anomaly")
    for ax in axs:
        ax.set_xlabel("Easting (km, UTM 43N)"); ax.set_aspect("equal")
        ax.plot(MAIN_HIGH[0] / 1e3, MAIN_HIGH[1] / 1e3, "+", color="k", ms=8); ax.plot(NW_HIGH[0] / 1e3, NW_HIGH[1] / 1e3, "x", color="k", ms=6)
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "terrain.png"); plt.close(fig)

    # ── data and regional field ──
    gx, gy, g_res = gridded(d["locations"], d["observed"])
    _, _, g_reg = gridded(d["locations"], d["regional"])
    _, _, g_raw = gridded(d["locations"], d["observed"] + d["regional"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"wspace": 0.45})
    v = np.nanmax(np.abs(g_res))
    for ax, g, tt, cm, lim in [(axs[0], g_raw, "Complete Bouguer anomaly (1 km)", "viridis", None),
                               (axs[1], g_reg, "Second-order trend surface (regional)", "viridis", None),
                               (axs[2], g_res, "Residual anomaly (inverted)", "RdBu_r", (-v, v))]:
        im = imshow(ax, g, gx, gy, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        ax.set_title(tt); ax.set_xlabel("Easting (km, UTM 43N)"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78, label="mGal")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "data.png"); plt.close(fig)

    def draw_section(ax, g, title):
        x, z, v, ground, northing = g.section(MAIN_HIGH[1])
        im = ax.pcolormesh(x, z, v, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
        ax.axhline(g.z_core_base / 1e3, color="k", lw=0.7, ls="--")
        ax.axvline(MAIN_HIGH[0] / 1e3, color="k", lw=0.5, ls=":")
        ax.set_ylim(z.min(), 1.6); ax.set_ylabel("Elevation (km)")
        ax.set_title(f"{title} — northing {northing / 1e3:.1f} km", loc="left")
        return im

    # ── what the terrain changes: four runs at β = 1 ──
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.1, 1], wspace=0.14, hspace=0.42)
    for r, (k, lab) in enumerate(STEPS):
        ax = fig.add_subplot(gs[r, 0])
        im = draw_section(ax, step_grids[k], lab)
        if r == 3:
            ax.set_xlabel("Easting (km, UTM 43N)")
        ax2 = fig.add_subplot(gs[r, 1])
        g = step_grids[k]
        x, y, vv, dk = g.depth_slice(1.0)
        ax2.pcolormesh(x, y, vv.T, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax2.set_aspect("equal"); ax2.tick_params(labelsize=7)
        ax2.set_title(f"{dk:.1f} km below the mean ground", fontsize=8.5)
    fig.colorbar(im, ax=fig.axes, shrink=0.45, label="Density contrast (g/cc)")
    fig.savefig(FIGS / "steps.png"); plt.close(fig)

    # the top kilometre with and without the correction
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"wspace": 0.3})
    g = grids["as1_beta1"]
    xe = g.mesh.nodes_x[g.n_pad:g.shape[0] - g.n_pad + 1] / 1e3
    ye = g.mesh.nodes_y[g.n_pad:g.shape[1] - g.n_pad + 1] / 1e3
    for ax, vv, tt, lim in [(axs[0], shallow["no_tc"], "No terrain correction", 0.3),
                            (axs[1], shallow["tc"], "With the terrain correction", 0.3),
                            (axs[2], shallow["tc"] - shallow["no_tc"], "Difference", 0.15)]:
        im = ax.pcolormesh(xe, ye, vv.T, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_aspect("equal"); ax.set_title(tt); ax.set_xlabel("Easting (km, UTM 43N)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="g/cc·km")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "shallow.png"); plt.close(fig)

    # ── residual maps of the six runs ──
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 8), gridspec_kw={"wspace": 0.25, "hspace": 0.3})
    for ax, (k, lab, *_) in zip(axs.ravel(), FULL):
        dd = runs[k]["_data"]
        x, y, g = gridded(dd["locations"], dd["observed"] - dd["predicted"])
        im = imshow(ax, g, x, y, cmap="RdBu_r", vmin=-3, vmax=3)
        f = numbers["full"][k]
        ax.set_title(f"{lab}\nχ²/N {f['chi2']:.2f} · RMS {f['rms']:.2f} mGal", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axs, shrink=0.6, label="Residual, observed − predicted (mGal)")
    fig.savefig(FIGS / "residuals.png"); plt.close(fig)

    # ── depth slices ──
    fig, axs = plt.subplots(len(SLICE_RUNS), 3, figsize=(10, 3.0 * len(SLICE_RUNS)), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.08, "hspace": 0.25})
    for r, k in enumerate(SLICE_RUNS):
        lab = next(l for kk, l, *_ in FULL if kk == k)
        for c, depth_km in enumerate([2.0, 5.0, 8.0]):
            x, y, vv, dk = grids[k].depth_slice(depth_km)
            ax = axs[r, c]
            im = ax.pcolormesh(x, y, vv.T, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
            ax.set_title((lab if c == 0 else "") + f"\n{dk:.1f} km below the mean ground", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=axs, shrink=0.4, label="Density contrast (g/cc)")
    fig.savefig(FIGS / "slices.png"); plt.close(fig)

    # ── E-W sections through the main high ──
    fig, axs = plt.subplots(len(FULL), 1, figsize=(10, 1.95 * len(FULL)), sharex=True)
    for ax, (k, lab, *_) in zip(axs, FULL):
        im = draw_section(ax, grids[k], lab)
    axs[-1].set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im, ax=axs, shrink=0.5, label="Density contrast (g/cc)")
    fig.savefig(FIGS / "sections.png"); plt.close(fig)

    # ── robust (integrated) vs not robust (centroid depth) ──
    fig, axs = plt.subplots(2, len(SLICE_RUNS), figsize=(16, 6.4), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.08, "hspace": 0.2})
    for c, k in enumerate(SLICE_RUNS):
        a = axs[0, c].pcolormesh(xe, ye, integ[k].T, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        b = axs[1, c].pcolormesh(xe, ye, cent[k].T, cmap="viridis_r", vmin=0, vmax=15)
        axs[0, c].set_title(next(l for kk, l, *_ in FULL if kk == k), fontsize=8.5)
        for ax in axs[:, c]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.colorbar(a, ax=axs[0], shrink=0.8, label="Vertically integrated density (g/cc·km)")
    fig.colorbar(b, ax=axs[1], shrink=0.8, label="Centroid depth below the ground (km)")
    fig.savefig(FIGS / "robust.png"); plt.close(fig)

    # ── |mass| with depth ──
    edges = np.arange(0, 22.01, 0.5)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for k, lab, colr, ls in FULL:
        axs[0].plot(grids[k].mass_profile(edges) * 100, mid, ls, color=colr, lw=1.8, label=lab)
    ledges = np.arange(0, 22.01, 1.0)
    for k, lab, colr in LOW_PROFILES:
        if k in lgrids:
            axs[1].plot(lgrids[k].mass_profile(ledges) * 100, (ledges[:-1] + ledges[1:]) / 2, "-", color=colr, lw=1.8, label=lab)
    for ax, tt in zip(axs, ["Full resolution (1 km mesh)", "2 km study"]):
        ax.axhline(10, color="k", lw=0.8, ls="--"); ax.set_title(tt)
        ax.set_xlabel("|mass| share (%/km, below the core area)")
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7.5)
    axs[0].set_ylabel("Depth below the ground (km)"); axs[0].set_ylim(22, 0)
    fig.savefig(FIGS / "mass_profiles.png"); plt.close(fig)

    # ── density under the main high: with terrain (solid) and the flat-earth runs (faint) ──
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, xy, tt in [(axs[0], MAIN_HIGH, "Under the main Bouguer high"), (axs[1], NW_HIGH, "Under the north-western high")]:
        for k, lab, colr, ls in FULL:
            z, vv, _ = grids[k].column(xy)
            ax.plot(vv, z, ls, color=colr, lw=1.8, label=lab)
            z, vv, _ = ogrids[k].column(xy)
            ax.plot(vv, z, ls, color=colr, lw=0.9, alpha=0.45)
        ax.axvline(0, color="k", lw=0.5); ax.set_ylim(22, 0); ax.grid(alpha=0.3)
        ax.set_xlabel("Density contrast (g/cc)"); ax.set_title(tt)
    axs[0].set_ylabel("Depth below the ground (km)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower right", title="thick: with terrain · thin: flat earth (28 Sep)",
                  title_fontsize=7.5)
    fig.savefig(FIGS / "centre_profiles.png"); plt.close(fig)

    # ── flat earth against terrain: depth range of the main body ──
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, (k, lab, *_) in enumerate(FULL):
        for off, src, colr, name in [(-0.16, numbers["old"][k]["main"], "#9aa7b3", "Flat earth, no terrain correction (28 Sep)"),
                                     (0.16, numbers["full"][k]["main"], "#1f5f8b", "Terrain and terrain correction")]:
            ax.plot([i + off, i + off], [src["top_km"], src["bottom_km"]], "-", color=colr, lw=8, solid_capstyle="butt",
                    label=name if i == 0 else None)
            ax.plot(i + off, src["centroid_km"], "o", color="white", ms=4.5, mec="k", mew=0.6)
    ax.set_xticks(range(len(FULL)))
    ax.set_xticklabels(["original", "L1–L2", "β=0.5", "β=1", "β=1.5", "α_s=0.1"])
    ax.set_ylim(21, 0); ax.set_ylabel("Depth below the ground (km)")
    ax.set_title("Half-maximum depth range (bars) and centroid (dots) under the main high", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGS / "old_vs_new.png"); plt.close(fig)

    # ── the 2 km study against the 1 km runs ──
    pairs = [("β=0.5", "as1_p0222_dw0.5", "as1_beta0.5"), ("β=1", "as1_p0222_dw1", "as1_beta1"),
             ("β=1.5", "as1_p0222_dw1.5", "as1_beta1.5"), ("α_s=0.1, β=1", "as0.1_p0222_dw1", "as0.1_beta1")]
    if all(lk in numbers["low"] for _, lk, _ in pairs):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (lab, lk, fk) in enumerate(pairs):
            for off, src, colr, name in [(-0.15, numbers["low"][lk]["main"], "#9aa7b3", "2 km mesh (1,296 data)"),
                                         (0.15, numbers["full"][fk]["main"], "#1f5f8b", "1 km mesh (5,040 data)")]:
                ax.plot([i + off, i + off], [src["top_km"], src["bottom_km"]], "-", color=colr, lw=7, solid_capstyle="butt",
                        label=name if i == 0 else None)
                ax.plot(i + off, src["centroid_km"], "o", color="white", ms=4, mec="k", mew=0.6)
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[0] for p in pairs])
        ax.set_ylim(12, 0); ax.set_ylabel("Depth below the ground (km)")
        ax.set_title("Half-maximum depth range under the main high: 2 km study and 1 km runs", fontsize=9.5)
        ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.savefig(FIGS / "lowres_vs_full.png"); plt.close(fig)

    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print("figures and numbers written to", FIGS)


if __name__ == "__main__":
    main()
