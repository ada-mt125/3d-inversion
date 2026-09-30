"""The 2 km runs with the density bounds from the rock samples, against the +0.5 / -0.2 bounds.

    py examples/output/karnataka_joint/scripts/compare_bounds.py

Reads data/lowres_runs_fixed/<coupling> (bounds -0.2 / +0.5) and data/lowres_bounds/<key>/
<coupling> (joint_params.DENSITY_BOUNDS), writes data/lowres_bounds/summary.json and prints a
table: the fit, how much of the dense model sits at a bound, the column under the main high
(its mass, and the thickness that mass needs at the measured +0.3 contrast), whether the belt
reaches the ground, and the coupling measures; and two figures there: the density on the E-W
section through the belt (sections.png) and the column under the main high (column.png).
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
from joint_params import DENSITY_BOUNDS, GRAVITY_REG   # noqa: E402
from kmodel import MAIN_HIGH, Grid, body, box_stats, data_fit   # noqa: E402
from make_figures import BOX, RHO, SANDUR_N, coupling_measures, load_joint   # noqa: E402

LOW = ROOT / "data" / "lowres_runs_fixed"
BOUNDS = ROOT / "data" / "lowres_bounds"
COUPLINGS = ("none", "joint_total_variation")
CONTRAST = 0.3          # metabasalt minus granite, measured (density_report.md)
KEEL_KM = 6.0           # the published base of the Sandur belt (Maurya et al. 2023)


def measures(path, lo, hi):
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
        "bounds": [lo, hi],
        "gravity_chi2": data_fit(r["_datas"]["gravity"])["chi2"],
        "magnetics_chi2": data_fit(r["_datas"]["magnetics"])["chi2"],
        "dense_at_upper": float((rho[dense] >= hi - 0.01).mean()),
        "cells_at_lower": int((rho <= lo + 0.01).sum()),
        "n_dense": int(dense.sum()),
        "main": {**body(z, v, dz), "mass": mass, "thickness_at_contrast": mass / CONTRAST,
                 "top_km_mean": float(v[near].mean())},
        "box_dense_at_ground": float((top_mean[box] > 0.1).mean()),
        "box": box_stats(Grid(r, np.clip(r["_models"]["gravity"], 0, None)), BOX),
        "magnetic_box": box_stats(gm, BOX),
        "coupling": coupling_measures(gg, gm),
        "shares": gg.shares(),
    }


def figures(paths):
    """The E-W section through the belt and the column under the main high, per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["rho50"] + [k for k in sorted(DENSITY_BOUNDS, key=lambda k: -DENSITY_BOUNDS[k][1])]
    label = {"none": "no coupling", "joint_total_variation": "JTV"}
    bounds = {"rho50": (GRAVITY_REG["bounds_lower"], GRAVITY_REG["bounds_upper"]), **DENSITY_BOUNDS}
    plt.rcParams.update({"font.size": 9})
    fig, axs = plt.subplots(len(COUPLINGS), len(keys), figsize=(3.6 * len(keys), 2.4 * len(COUPLINGS)),
                            sharex=True, sharey=True, layout="constrained", squeeze=False)
    fig2, axs2 = plt.subplots(1, len(COUPLINGS), figsize=(4.2 * len(COUPLINGS), 4.2), sharey=True,
                              layout="constrained", squeeze=False)
    im = None
    for r_, c in enumerate(COUPLINGS):
        for k_, key in enumerate(keys):
            path = paths.get(f"{key}/{c}")
            ax = axs[r_, k_]
            if path is None:
                ax.set_visible(False)
                continue
            _, gg, _ = load_joint(path)
            x, z, v, ground, _ = gg.section(SANDUR_N)
            im = ax.pcolormesh(x, z, v, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            ax.plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.8)
            ax.set_ylim(-10.5, 1.6)
            lo, hi = bounds[key]
            ax.set_title(f"{label[c]} · bounds {lo:+.2f} / {hi:+.2f}", loc="left", fontsize=9)
            zc, vc, _ = gg.column(MAIN_HIGH)
            axs2[0, r_].step(vc, zc, where="mid", label=f"{hi:+.2f}")
        axs2[0, r_].set_title(f"{label[c]}: under the main high", loc="left", fontsize=9)
        axs2[0, r_].axhline(KEEL_KM, color="k", lw=0.7, ls="--")
        axs2[0, r_].set_xlabel("Density contrast (g/cc)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Elevation (km)")
    for ax in axs[-1]:
        ax.set_xlabel("Easting (km)")
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.5, label="Density contrast (g/cc)", location="bottom", aspect=40)
    axs2[0, 0].set_ylim(12, 0)
    axs2[0, 0].set_ylabel("Depth below the ground (km)")
    axs2[0, 0].legend(title="upper bound", fontsize=8)
    fig.savefig(BOUNDS / "sections.png", dpi=150)
    fig2.savefig(BOUNDS / "column.png", dpi=150)
    plt.close(fig)
    plt.close(fig2)


def main():
    runs, paths = {}, {}
    base = (GRAVITY_REG["bounds_lower"], GRAVITY_REG["bounds_upper"])
    for c in COUPLINGS:
        if (LOW / c / "result.zip").exists():
            runs[f"rho50/{c}"] = measures(LOW / c, *base)
            paths[f"rho50/{c}"] = LOW / c
        for key, (lo, hi) in DENSITY_BOUNDS.items():
            if (BOUNDS / key / c / "result.zip").exists():
                runs[f"{key}/{c}"] = measures(BOUNDS / key / c, lo, hi)
                paths[f"{key}/{c}"] = BOUNDS / key / c
    BOUNDS.mkdir(parents=True, exist_ok=True)
    (BOUNDS / "summary.json").write_text(json.dumps(runs, indent=1))
    head = (f"{'run':32s} {'chi2 g/m':>10s} {'@upper':>7s} {'@lower':>7s} {'M':>5s} {'M/0.3':>6s} "
            f"{'top-bot km':>11s} {'cen':>5s} {'0-1km':>6s} {'box@gnd':>8s} {'d10-d90':>10s} "
            f"{'edges':>6s} {'mag∈ρ':>6s}")
    print(head)
    for name, m in runs.items():
        mn, b, c = m["main"], m["box"], m["coupling"]
        print(f"{name:32s} {m['gravity_chi2']:4.2f}/{m['magnetics_chi2']:4.2f} "
              f"{m['dense_at_upper']:7.0%} {m['cells_at_lower']:7d} {mn['mass']:5.2f} "
              f"{mn['thickness_at_contrast']:6.1f} {mn['top_km']:5.2f}-{mn['bottom_km']:<5.2f} "
              f"{mn['centroid_km']:5.2f} {mn['top_km_mean']:6.3f} {m['box_dense_at_ground']:8.0%} "
              f"{b['d10_km']:4.1f}-{b['d90_km']:<5.1f} {c['edges_shared']:6.0%} "
              f"{c['support']['magnetic_in_dense']:6.0%}")
    print(f"keel {KEEL_KM} km; M/0.3 = the thickness the column's mass needs at +{CONTRAST} g/cc")
    figures(paths)


if __name__ == "__main__":
    main()
