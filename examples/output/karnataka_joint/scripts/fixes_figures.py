"""Figures of the follow-up to Section 5 (the fixes of 30 September, evening).

    py examples/output/karnataka_joint/scripts/fixes_figures.py [--low]

fixes_sections.png: E-W sections through the Sandur belt of the group lasso as reported, the
group lasso fixed, and the uncoupled run (fixed), down into the bottom padding.
fixes_mvi.png: the magnetic residuals of the induced single inversion (beta1) and of the
magnetization-vector inversion (beta1_mvi), and the MVI amplitude integrated with depth.
--low: the 2 km runs (data/lowres_runs, data/lowres_runs_fixed; no MVI panel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from make_figures import CHI_TICKS, FIGS, RHO, SANDUR_N, chi_style, imshow, load_joint  # noqa: E402
from fixes_summary import MAGNETIC, UNDERFIT_NT  # noqa: E402
from kmodel import Grid, data_fit, gridded, load  # noqa: E402

LOW = "--low" in sys.argv
BEFORE = ROOT / "data" / ("lowres_runs" if LOW else "ec2_runs")
AFTER = ROOT / "data" / ("lowres_runs_fixed" if LOW else "ec2_runs_fixed")
ROWS = [(BEFORE / "group_lasso", "Group lasso as reported (paper settings)"),
        (AFTER / "group_lasso_depth", "Group lasso fixed (depth weighting, balanced)"),
        (AFTER / "none", "No coupling (SimPEG sparse)")]
SUFFIX = "_2km" if LOW else ""


def sections():
    rows = [(p, lab) for p, lab in ROWS if (p / "result.zip").exists()]
    fig, axs = plt.subplots(len(rows), 2, figsize=(14, 2.3 * len(rows)), sharex=True,
                            layout="constrained", squeeze=False)
    for r_, (path, label) in enumerate(rows):
        _, gg, gm = load_joint(path)
        for c, (g, style) in enumerate(((gg, RHO), (gm, chi_style()))):
            x, z, v, ground, _ = g.section(SANDUR_N)
            im = axs[r_, c].pcolormesh(x, z, v, **style)
            axs[r_, c].plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
            axs[r_, c].axhline(g.z_core_base / 1e3, color="k", lw=0.7, ls="--")
            axs[r_, c].set_ylim(g.mesh.nodes_z[0] / 1e3, 1.6)
            share = g.shares()
            axs[r_, c].set_title(f"{label} · {'density contrast' if c == 0 else 'susceptibility'} · "
                                 f"{share['core']:.0%} in the core, {share['below']:.0%} below",
                                 loc="left", fontsize=9)
            if c == 0:
                axs[r_, c].set_ylabel("Elevation (km)")
                im_r = im
            else:
                im_c = im
    for ax in axs[-1]:
        ax.set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im_r, ax=axs[:, 0], shrink=0.5, label="Density contrast (g/cc)", location="bottom", aspect=40)
    fig.colorbar(im_c, ax=axs[:, 1], shrink=0.5, label="Susceptibility (SI)", ticks=CHI_TICKS,
                 location="bottom", aspect=40)
    fig.savefig(FIGS / f"fixes_sections{SUFFIX}.png")
    plt.close(fig)


def mvi():
    if LOW or not (MAGNETIC / "beta1_mvi" / "result.zip").exists():
        return
    a, b = load(MAGNETIC / "beta1"), load(MAGNETIC / "beta1_mvi")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    bad = (a["_data"]["observed"] - a["_data"]["predicted"]) > UNDERFIT_NT
    for ax, r, title in ((axs[0], a, "Induced (susceptibility, β = 1)"),
                         (axs[1], b, "Magnetization vector (MVI, β = 1)")):
        d = r["_data"]
        x, y, g = gridded(d["locations"], d["observed"] - d["predicted"])
        im = imshow(ax, g, x, y, cmap="RdBu_r", vmin=-150, vmax=150)
        fit = data_fit(d)
        ax.plot(d["locations"][bad, 0] / 1e3, d["locations"][bad, 1] / 1e3, "k.", ms=2.5)
        ax.set_title(f"{title}\nχ²/N {fit['chi2']:.2f} · RMS {fit['rms']:.0f} nT · max {fit['max_abs']:.0f} nT",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km)")
    axs[0].set_ylabel("Northing (km)")
    fig.colorbar(im, ax=axs[:2], shrink=0.8, label="magnetic residual (nT)", extend="both")
    g = Grid(b)
    total, _ = g.integrated()
    xe = g.mesh.nodes_x[g.n_pad:g.shape[0] - g.n_pad + 1] / 1e3
    ye = g.mesh.nodes_y[g.n_pad:g.shape[1] - g.n_pad + 1] / 1e3
    im2 = axs[2].pcolormesh(xe, ye, total.T, cmap="Oranges", vmin=0, vmax=1.0)
    axs[2].plot(a["_data"]["locations"][bad, 0] / 1e3, a["_data"]["locations"][bad, 1] / 1e3, "k.", ms=2.5)
    info = b.get("magnetization") or {}
    axs[2].set_title(f"MVI amplitude integrated with depth\nstrong cells: I {info.get('resultant_inclination', 0):.0f}°, "
                     f"D {info.get('resultant_declination', 0):.0f}° (field I {info.get('inducing_inclination', 0):.0f}°, "
                     f"D {info.get('inducing_declination', 0):.0f}°)", fontsize=9)
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("Easting (km)")
    fig.colorbar(im2, ax=axs[2], shrink=0.8, label="|m| × thickness (SI·km)", extend="max")
    fig.savefig(FIGS / "fixes_mvi.png")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    FIGS.mkdir(exist_ok=True)
    sections()
    mvi()
    print("figures in", FIGS)


if __name__ == "__main__":
    main()
