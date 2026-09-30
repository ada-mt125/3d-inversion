"""The 2 km runs with the density bounds from the rock samples, against the +0.5 / -0.2 bounds.

    py examples/output/karnataka_joint/scripts/compare_bounds.py [--keys rho50,rho35,rho35_gb05]

Reads data/lowres_runs_fixed/<coupling> (bounds -0.2 / +0.5, beta 1; "rho50") and
data/lowres_bounds/<key>/<coupling> (joint_params.VARIANTS), writes data/lowres_bounds/
summary.json and prints a table: the fit, how much of the dense model sits at a bound, the
column under the main high (its mass, and the thickness that mass needs at the measured +0.3
contrast), whether the belt reaches the ground, where the magnetic model lies, and the coupling
measures.  Figures there, for the runs of --keys (default: all): the density and the
susceptibility on the E-W section through the belt (sections.png, sections_chi.png) and the
density column under the main high (column.png).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from joint_params import GRAVITY_REG, MAGNETIC_REG, VARIANTS   # noqa: E402
from kmodel import MAIN_HIGH, Grid, body, box_stats, data_fit   # noqa: E402
from make_figures import BOX, SANDUR_N, chi_style, coupling_measures, load_joint   # noqa: E402

LOW = ROOT / "data" / "lowres_runs_fixed"
BOUNDS = ROOT / "data" / "lowres_bounds"
COUPLINGS = ("none", "joint_total_variation")
LABEL = {"none": "no coupling", "joint_total_variation": "JTV"}
CONTRAST = 0.3          # metabasalt minus granite, measured (density_report.md)
KEEL_KM = 6.0           # the base of the Sandur belt in Maurya et al. (2023, conference abstract)


def settings(key):
    """(lower, upper) density bounds and (gravity, magnetic) betas of a run."""
    v = VARIANTS.get(key, {})
    return (tuple(v.get("density_bounds", (GRAVITY_REG["bounds_lower"], GRAVITY_REG["bounds_upper"]))),
            tuple(v.get("betas", (GRAVITY_REG["depth_weighting_exponent"],
                                  MAGNETIC_REG["depth_weighting_exponent"]))))


def title(key):
    (lo, hi), (bg, bm) = settings(key)
    return f"{lo:+.2f} / {hi:+.2f}, β {bg:g} / {bm:g}"


def top_share(g, box, km=1.0):
    """Share of the |property| x volume in ``box`` that lies within ``km`` of the ground."""
    cx, cy = g.mesh.cell_centers_x, g.mesh.cell_centers_y
    sel = ((cx >= box[0]) & (cx <= box[1]))[:, None] & ((cy >= box[2]) & (cy <= box[3]))[None, :]
    a = np.clip(g.m, 0, None) * g.vol * g.active * sel[:, :, None]
    return float(a[g.depth < km * 1e3].sum() / a.sum())


def measures(path, key):
    (lo, hi), betas = settings(key)
    r, gg, gm = load_joint(path)
    core = gg.core_xy[:, :, None] & gg.active & (gg.zc >= gg.z_core_base)[None, None, :]
    rho = gg.m[core]
    dense = rho > 0.05
    z, v, dz = gg.column(MAIN_HIGH)
    mass = float((np.clip(v, 0, None) * dz).sum())
    near = z < 1.0                       # the cells within 1 km of the ground
    # in the Sandur box: the share of columns whose top kilometre is dense
    cx, cy = gg.mesh.cell_centers_x, gg.mesh.cell_centers_y
    box = ((cx >= BOX[0]) & (cx <= BOX[1]))[:, None] & ((cy >= BOX[2]) & (cy <= BOX[3]))[None, :]
    top = gg.active & (gg.depth < 1000.0)
    top_mean = np.where(top, gg.m, 0).sum(axis=2) / np.maximum(top.sum(axis=2), 1)
    return {
        "bounds": [lo, hi], "betas": list(betas),
        "gravity_chi2": data_fit(r["_datas"]["gravity"])["chi2"],
        "magnetics_chi2": data_fit(r["_datas"]["magnetics"])["chi2"],
        "dense_at_upper": float((rho[dense] >= hi - 0.01).mean()),
        "cells_at_lower": int((rho <= lo + 0.01).sum()),
        "n_dense": int(dense.sum()),
        "main": {**body(z, v, dz), "mass": mass, "thickness_at_contrast": mass / CONTRAST,
                 "top_km_mean": float(v[near].mean())},
        "box_dense_at_ground": float((top_mean[box] > 0.1).mean()),
        "box_dense_top1km": top_share(gg, BOX),
        "box": box_stats(Grid(r, np.clip(r["_models"]["gravity"], 0, None)), BOX),
        "magnetic_box": box_stats(gm, BOX),
        "box_magnetic_top1km": top_share(gm, BOX),
        "coupling": coupling_measures(gg, gm),
        "shares": gg.shares(),
        "magnetic_shares": gm.shares(),
    }


def figures(paths, keys):
    """The E-W sections through the belt and the column under the main high, per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    keys = [k for k in keys if any(f"{k}/{c}" in paths for c in COUPLINGS)]
    figs = {}
    for name in ("sections", "sections_chi"):
        figs[name] = plt.subplots(len(COUPLINGS), len(keys), figsize=(3.6 * len(keys), 2.4 * len(COUPLINGS)),
                                  sharex=True, sharey=True, layout="constrained", squeeze=False)
    fig2, axs2 = plt.subplots(1, len(COUPLINGS), figsize=(4.2 * len(COUPLINGS), 4.2), sharey=True,
                              layout="constrained", squeeze=False)
    ims = {}
    for r_, c in enumerate(COUPLINGS):
        for k_, key in enumerate(keys):
            path = paths.get(f"{key}/{c}")
            if path is None:
                for f_, a_ in figs.values():
                    a_[r_, k_].set_visible(False)
                continue
            _, gg, gm = load_joint(path)
            for name, g, style in (("sections", gg, dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5)),
                                   ("sections_chi", gm, chi_style())):
                ax = figs[name][1][r_, k_]
                x, z, v, ground, _ = g.section(SANDUR_N)
                ims[name] = ax.pcolormesh(x, z, v, **style)
                ax.plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.8)
                ax.set_ylim(-10.5, 1.6)
                ax.set_title(f"{LABEL[c]} · {title(key)}", loc="left", fontsize=8.5)
            zc, vc, _ = gg.column(MAIN_HIGH)
            axs2[0, r_].step(vc, zc, where="mid", label=title(key))
        axs2[0, r_].set_title(f"{LABEL[c]}: under the main high", loc="left", fontsize=9)
        axs2[0, r_].axhline(KEEL_KM, color="k", lw=0.7, ls="--")
        axs2[0, r_].set_xlabel("Density contrast (g/cc)")
    for name, unit in (("sections", "Density contrast (g/cc)"), ("sections_chi", "Susceptibility (SI)")):
        fig, axs = figs[name]
        for ax in axs[:, 0]:
            ax.set_ylabel("Elevation (km)")
        for ax in axs[-1]:
            ax.set_xlabel("Easting (km)")
        if name in ims:
            fig.colorbar(ims[name], ax=axs, shrink=0.5, label=unit, location="bottom", aspect=40)
        fig.savefig(BOUNDS / f"{name}.png", dpi=150)
        plt.close(fig)
    axs2[0, 0].set_ylim(12, 0)
    axs2[0, 0].set_ylabel("Depth below the ground (km)")
    axs2[0, 0].legend(title="density bounds, β gravity / magnetics", fontsize=7, title_fontsize=7)
    fig2.savefig(BOUNDS / "column.png", dpi=150)
    plt.close(fig2)


def main():
    args = sys.argv[1:]
    keys = ["rho50"] + list(VARIANTS)
    if args[:1] == ["--keys"]:
        keys = args[1].split(",")
    runs, paths = {}, {}
    for c in COUPLINGS:
        for key in ["rho50"] + list(VARIANTS):
            path = LOW / c if key == "rho50" else BOUNDS / key / c
            if (path / "result.zip").exists():
                runs[f"{key}/{c}"] = measures(path, key)
                paths[f"{key}/{c}"] = path
    BOUNDS.mkdir(parents=True, exist_ok=True)
    (BOUNDS / "summary.json").write_text(json.dumps(runs, indent=1))
    print(f"{'run':32s} {'chi2 g/m':>10s} {'@upper':>7s} {'M':>5s} {'top-bot km':>11s} {'cen':>5s} "
          f"{'ρ<1km':>6s} {'box ρ d10-d50':>13s} {'χ<1km':>6s} {'box χ d10-d50':>13s} "
          f"{'edges':>6s} {'mag∈ρ':>6s}")
    for name, m in runs.items():
        mn, b, mb, c = m["main"], m["box"], m["magnetic_box"], m["coupling"]
        print(f"{name:32s} {m['gravity_chi2']:4.2f}/{m['magnetics_chi2']:4.2f} "
              f"{m['dense_at_upper']:7.0%} {mn['mass']:5.2f} {mn['top_km']:5.2f}-{mn['bottom_km']:<5.2f} "
              f"{mn['centroid_km']:5.2f} {m['box_dense_top1km']:6.0%} {b['d10_km']:6.1f}-{b['d50_km']:<6.1f} "
              f"{m['box_magnetic_top1km']:6.0%} {mb['d10_km']:6.1f}-{mb['d50_km']:<6.1f} "
              f"{c['edges_shared']:6.0%} {c['support']['magnetic_in_dense']:6.0%}")
    print("M: g/cc·km under the main high; ρ<1km, χ<1km: share of the box's density / susceptibility "
          "within 1 km of the ground")
    figures(paths, keys)


if __name__ == "__main__":
    main()
