"""Figures of the regularization trials on the Block-8 window, into figures/.

    py examples/output/block8_regularization/scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import trials  # noqa: E402

FIGS = ROOT / "figures"
RUNS = trials.RUNS
W = trials.WINDOW
X0, Y0 = np.mean(W[:2]), np.mean(W[2:])
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "figure.dpi": 150,
                     "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, RULE = "#1b232c", "#56626e", "#d6dde3"
FIT, HOLD = "#7a8794", "#1f5f8b"
CHI = LinearSegmentedColormap.from_list("chi", ["#ffffff", "#fde9b5", "#f6b15c", "#d9572b", "#8f1d2c", "#3d0a1e"])
GROUPS = {
    "A": ["default", "lp0221_L1", "lp0111", "lp0000", "lp0222", "l2", "mgs", "tv", "l1l2_irls08"],
    "B": ["default", "beta1", "beta2", "beta3", "sens", "L1", "L6", "as0.1", "as10", "b0inf", "free",
          "f0.75", "f3", "notrend"],
}


def km(v, ref):
    return (np.asarray(v) - ref) / 1000.0


def scores(names):
    out = []
    for n in names:
        f = RUNS / n / "score.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def short(s):
    """A trial's label for the axes."""
    return s["label"].replace("sparse ", "").replace("depth weighting", "depth w.")


def fig_data():
    """The TMI in the window (every node; the fitted ones are every second) and the ground."""
    from geoinv3d.cloud.worker import grid_window
    m = 4500.0
    xs, ys, V = grid_window(str(trials.INPUTS / "tmi_window.tif"), trials.CRS,
                            [W[0] - m, W[1] + m, W[2] - m, W[3] + m])
    _, _, Vw = grid_window(str(trials.INPUTS / "tmi_window.tif"), trials.CRS, W)
    gx, gy = np.meshgrid(np.linspace(W[0] - m, W[1] + m, 221), np.linspace(W[2] - m, W[3] + m, 221))
    G = trials.surface()(gx, gy)
    box = (km([W[0], W[1], W[1], W[0], W[0]], X0), km([W[2], W[2], W[3], W[3], W[2]], Y0))
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    lim = float(np.nanpercentile(np.abs(Vw), 99))
    im = ax[0].pcolormesh(km(xs, X0), km(ys, Y0), V, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                          shading="nearest", rasterized=True)
    ax[0].plot(*box, color="k", lw=1)
    fig.colorbar(im, ax=ax[0], shrink=0.85, label="TMI (nT), clipped at ±%d" % round(lim, -2), extend="both")
    ax[0].set_title(f"(a) TMI, 37.5 m; in the window {np.nanmin(Vw):,.0f} to {np.nanmax(Vw):,.0f} nT")
    im = ax[1].pcolormesh(km(gx[0], X0), km(gy[:, 0], Y0), G, cmap="terrain", vmin=450, vmax=1100,
                          shading="nearest", rasterized=True)
    ax[1].contour(km(xs, X0), km(ys, Y0), V, levels=[-2000, 2000], colors=["#08306b", "#67000d"], linewidths=0.8)
    ax[1].plot(*box, color="k", lw=1)
    fig.colorbar(im, ax=ax[1], shrink=0.85, label="ground (m, GLO-90)")
    ax[1].set_title("(b) the ground, with the ±2,000 nT contours of (a)")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel(f"east of {X0:,.0f} (km)")
    ax[0].set_ylabel(f"north of {Y0:,.0f} (km)")
    fig.savefig(FIGS / "data.png")
    plt.close(fig)


def fig_scores(group, name):
    """Misfit on the fitted and the left-out nodes; depth of the model; where it sits."""
    ss = scores(GROUPS[group])
    if not ss:
        return
    ss = ss[::-1]
    yy = np.arange(len(ss))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 0.36 * len(ss) + 1.9), sharey=True, constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    below_axes = dict(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=7,
                      handlelength=1, columnspacing=1)
    a = ax[0]
    for i, s in enumerate(ss):
        a.plot([s["chi2_fit_auto"], s["chi2_holdout"]], [i, i], color=RULE, lw=2, zorder=1)
    a.scatter([s["chi2_fit_auto"] for s in ss], yy, s=26, facecolor="white", edgecolor=FIT, lw=1.3, zorder=2,
              label="nodes fitted")
    a.scatter([s["chi2_holdout"] for s in ss], yy, s=26, color=HOLD, zorder=3, label="nodes left out")
    a.axvline(1, color=MUTED, lw=0.6, ls=":")
    hi = max(max(s["chi2_holdout"] for s in ss), 1.2)
    a.set_xlim(min(0.8, min(s["chi2_fit_auto"] for s in ss) - 0.05), hi * 1.05)
    a.set_xlabel("χ²/N, errors 2 % + 1.5 % of the spread")
    a.set_yticks(yy, [short(s) for s in ss])
    a.legend(ncol=2, **below_axes)
    a.set_title("(a) fit, and the nodes left out")
    a = ax[1]
    bands = [("share_0_200", "0–200 m", "#3d0a1e"), ("share_200_600", "200–600 m", "#d9572b"),
             ("share_600_1500", "600–1,500 m", "#f6b15c"), ("share_1500_9999", "> 1,500 m", "#d6dde3")]
    left = np.zeros(len(ss))
    for key, lab, col in bands:
        v = np.array([s[key] for s in ss])
        a.barh(yy, v, left=left, color=col, height=0.62, label=lab, edgecolor="white", lw=0.6)
        left += v
    a.set_xlim(0, 1)
    a.set_xlabel("share of |χ| × volume under the window")
    a.legend(ncol=4, **below_axes)
    a.set_title("(b) depth below the ground")
    a = ax[2]
    a.barh(yy, [s["beside"] for s in ss], color="#9aa7b3", height=0.62, label="beside the window", edgecolor="white", lw=0.6)
    a.barh(yy, [s["below"] for s in ss], left=[s["beside"] for s in ss], color="#1f5f8b", height=0.62,
           label="below the core (4 km)", edgecolor="white", lw=0.6)
    a.set_xlim(0, 1)
    a.set_xlabel("share of |χ| × volume of the whole model")
    a.legend(ncol=2, **below_axes)
    a.set_title("(c) outside the core")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


def section(name, half=3200.0, step=37.5):
    """The model along the NE-SW line through the window's centre (across the strike):
    distance (km, + to the NE), elevation of each depth sample, values."""
    z = np.load(RUNS / name / "model_grid.npz")
    xs, ys, g, depths, m = z["x"], z["y"], z["ground"], z["depths"], z["model"]
    t = np.arange(-half, half + step, step)
    px, py = X0 + t * np.sin(np.radians(45)), Y0 + t * np.cos(np.radians(45))
    ok = (px >= xs[0]) & (px <= xs[-1]) & (py >= ys[0]) & (py <= ys[-1])
    t, px, py = t[ok], px[ok], py[ok]
    ix = np.clip(np.round((px - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((py - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    vals = m[:, iy, ix]
    ground = g[iy, ix]
    elev = ground[None, :] - depths[:, None]
    return t / 1000.0, elev, vals, ground


def draw_section(a, name, title, vmax=0.5, zmin=-1200):
    t, elev, vals, ground = section(name)
    T = np.broadcast_to(t, elev.shape)
    im = a.pcolormesh(T, elev, np.clip(vals, 1e-4, None), cmap=CHI, norm=LogNorm(1e-3, vmax), shading="nearest",
                      rasterized=True)
    a.plot(t, ground, color=INK, lw=0.8)
    a.set_ylim(zmin, 1100)
    a.set_title(title, fontsize=8)
    return im


def fig_sections(names, fname, ncols=4, true=None):
    ss = {s["name"]: s for s in scores(names)}
    names = [n for n in names if n in ss]
    if true:
        names = [true] + names
    nrows = int(np.ceil(len(names) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(10.5, 2.05 * nrows + 0.6), sharex=True, sharey=True,
                            constrained_layout=True)
    axs = np.atleast_2d(axs)
    im = None
    for k, a in enumerate(axs.flat):
        if k >= len(names):
            a.axis("off")
            continue
        n = names[k]
        if n == true:
            im = draw_section(a, n, "true model")
        else:
            s = ss[n]
            im = draw_section(a, n, f"{short(s)}\nχ²/N {s['chi2_fit_auto']:.2f} / left out {s['chi2_holdout']:.2f}")
        if k % ncols == 0:
            a.set_ylabel("elevation (m)")
        if k // ncols == nrows - 1:
            a.set_xlabel("distance NE of the centre (km)")
    fig.colorbar(im, ax=axs, shrink=0.6, label="susceptibility (SI)", extend="both")
    fig.savefig(FIGS / f"{fname}.png")
    plt.close(fig)


def true_grid():
    """The synthetic model on the figures' grid, as RUNS/_true/model_grid.npz."""
    out = RUNS / "_true" / "model_grid.npz"
    if not (trials.INPUTS / "synthetic_model.npy").exists():
        return
    src = next(n for n in ("default", json.loads((trials.INPUTS / "synthetic.json").read_text())["from_trial"])
               if (RUNS / n / "result.zip").exists())
    r = trials.load(src)   # every trial has the same mesh
    xs, ys, g, mg = trials.model_grid(trials.mesh_of(r["meta"]), r["active"],
                                      np.load(trials.INPUTS / "synthetic_model.npy"), trials.surface())
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, x=xs, y=ys, ground=g, depths=trials.GRID_DEPTHS, model=mg)


def fig_slices(names, fname, depths=(100.0, 400.0, 900.0), true=None):
    """Plan views of the model at a few depths below the ground."""
    ss = {s["name"]: s for s in scores(names)}
    names = ([true] if true else []) + [n for n in names if n in ss]
    fig, axs = plt.subplots(len(depths), len(names), figsize=(1.55 * len(names) + 0.9, 1.6 * len(depths) + 0.5),
                            sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    im = None
    for j, n in enumerate(names):
        z = np.load(RUNS / n / "model_grid.npz")
        for i, d in enumerate(depths):
            k = int(np.argmin(np.abs(z["depths"] - d)))
            a = axs[i, j]
            im = a.pcolormesh(km(z["x"], X0), km(z["y"], Y0), np.clip(z["model"][k], 1e-4, None), cmap=CHI,
                              norm=LogNorm(1e-3, 0.5), shading="nearest", rasterized=True)
            a.set_aspect("equal")
            a.tick_params(labelsize=6.5)
            if i == 0:
                a.set_title("true model" if n == true else short(ss[n]), fontsize=7.5)
            if j == 0:
                a.set_ylabel(f"{z['depths'][k]:.0f} m deep", fontsize=7.5)
    fig.colorbar(im, ax=axs, shrink=0.7, label="susceptibility (SI)", extend="both")
    fig.savefig(FIGS / f"{fname}.png")
    plt.close(fig)


SYN = ["syn-default", "syn-sens", "syn-lp0221_L1", "syn-beta1", "syn-beta2", "syn-L1", "syn-l2",
       "syn-l1l2_irls08", "syn-mgs"]


def main():
    FIGS.mkdir(exist_ok=True)
    fig_data()
    fig_scores("A", "scores_regularizations")
    fig_scores("B", "scores_settings")
    fig_sections(GROUPS["A"], "sections_regularizations")
    fig_sections(GROUPS["B"], "sections_settings")
    fig_slices(["default", "sens", "lp0221_L1", "beta1", "l2", "l1l2_irls08"], "slices")
    true_grid()
    if (RUNS / "_true" / "model_grid.npz").exists():
        GROUPS["S"] = SYN
        fig_sections(SYN, "sections_synthetic", true="_true")
        fig_slices(SYN[:6], "slices_synthetic", true="_true")
        fig_scores("S", "scores_synthetic")
    print("figures in", FIGS)


if __name__ == "__main__":
    main()
