"""Why the group lasso did best and the cross-gradient did little: two figures per shape.

    py examples/output/coupling_comparison/make_why_figure.py [blocks] [dipping]

1. Sections of L1–L2 alone (IRLS), the group lasso's solver with no coupling (the control),
   and the group lasso: what the solver does and what the coupling adds.
2. The cross-gradient's view of the uncoupled L1–L2 models: sin θ between ∇Δρ and ∇χ where
   both vary, and where only the density varies (the cross-gradient is zero there whatever
   the density does).  Writes report/<shape>_5_why.png, report/<shape>_6_crossgradient.png and
   <shape>/why.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import make_report_figures as F   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
from geoinv3d.viz.result_workflow import load_result, result_mesh   # noqa: E402

ROWS = [("none", "1. L1–L2 solved by IRLS, no coupling"),
        ("group_lasso_uncoupled", "2. L1 + L2 solved by ADMM, no coupling"),
        ("group_lasso", "3. Group lasso: L1 on each cell's pair, ADMM")]
CM_SIN = LinearSegmentedColormap.from_list("sin", ["#f3f7fc", "#9ec5f4", "#3987e5", "#184f95"])


def full_grid(run, values):
    mesh = result_mesh(run)
    full = np.zeros(mesh.n_cells)
    act = run.get("_active")
    if act is not None:
        full[np.asarray(act, dtype=bool)] = values
    else:
        full[:] = values
    return mesh, full.reshape(mesh.shape_cells, order="F")


def gradients(mesh, grid):
    gx, gy, gz = np.gradient(grid, mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z)
    return np.stack([gx, gy, gz], axis=-1)


def crossgradient_view(run):
    """sin θ between the models' gradients on the section, and the cells where only one varies."""
    mesh, rho = full_grid(run, run["_models"]["gravity"])
    _, chi = full_grid(run, run["_models"]["magnetics"])
    gr, gc = gradients(mesh, rho), gradients(mesh, chi)
    nr, nc = np.linalg.norm(gr, axis=-1), np.linalg.norm(gc, axis=-1)
    cx, cy, cz = mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z
    core = ((cx[:, None, None] > -50) & (cx[:, None, None] < 1550) & (cy[None, :, None] > -50)
            & (cy[None, :, None] < 1250) & (cz[None, None, :] > -600))
    sr = nr > 0.1 * nr[core].max()
    sc = nc > 0.1 * nc[core].max()
    both = core & sr & sc
    sin = np.full(nr.shape, np.nan)
    cross = np.linalg.norm(np.cross(gr, gc), axis=-1)
    sin[both] = cross[both] / (nr[both] * nc[both])
    only_rho = core & sr & ~sc
    j = int(np.argmin(np.abs(cy - F.Y0)))
    keep_x = (cx > -150) & (cx < 1650)
    keep_z = (cz < 0) & (cz > -650)
    sec = sin[:, j, :][np.ix_(keep_x, keep_z)].T
    only = only_rho[:, j, :][np.ix_(keep_x, keep_z)].T
    stats = {"cells_both_vary": int(both.sum()), "cells_only_density_varies": int(only_rho.sum()),
             "median_sin": float(np.nanmedian(sin[both])) if both.any() else None,
             "share_sin_below_0.3": float(np.mean(sin[both] < 0.3)) if both.any() else None}
    return cx[keep_x], -cz[keep_z], sec, only, sin[both], stats


def why_figure(shape, runs, table):
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(11, 1.95 * len(ROWS) + 1.1), sharex=True, sharey=True,
                             layout="constrained")
    for r, (key, title) in enumerate(ROWS):
        run = runs[key]
        for c, (model, cmap, vmax) in enumerate((("magnetics", F.CM_CHI, 0.05), ("gravity", F.CM_RHO, F.RHO))):
            x, d, sec = F.section(run, run["_models"][model])
            prop = "susceptibility" if model == "magnetics" else "density contrast"
            im = F.draw_section(axes[r, c], x, d, sec, cmap, vmax, shape, f"{title} · {prop}")
            fmt = (lambda v: f"{v:.4f}") if model == "magnetics" else (lambda v: f"{v:.3f}")
            b = table[key]["bodies"]
            vals = "   ".join(f"{n} {fmt(b[n]['chi_mean' if model == 'magnetics' else 'rho_mean'])}" for n in "ABC")
            axes[r, c].text(0.99, 0.04, f"body means  {vals}", transform=axes[r, c].transAxes, ha="right",
                            va="bottom", fontsize=7.5, color=F.INK,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            if c == 0:
                axes[r, c].set_ylabel("depth (m)")
            if r == len(ROWS) - 1:
                axes[r, c].set_xlabel("x (m)")
                cb = fig.colorbar(im, ax=axes[:, c], location="bottom", shrink=0.9, aspect=40)
                cb.set_label("susceptibility (SI)" if c == 0 else "density contrast (g/cc)", color=F.INK2)
                cb.outline.set_visible(False)
    fig.suptitle("Rows 1-2: one regularization, two solvers.  Rows 2-3: one solver, with and without pairing (the coupling)", x=0.01, ha="left",
                 fontsize=10, color=F.INK)
    path = F.OUT / f"{shape}_5_why.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def crossgradient_figure(shape, run):
    x, d, sec, only, sins, stats = crossgradient_view(run)
    fig = plt.figure(figsize=(11, 3.3), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0])
    ax.pcolormesh(x, d, np.where(only, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("g", ["#d9d7cf", "#d9d7cf"]),
                  shading="nearest", rasterized=True)
    im = ax.pcolormesh(x, d, sec, cmap=CM_SIN, vmin=0, vmax=1, shading="nearest", rasterized=True)
    for name, _, xc, _ in F.BODIES:
        ox, oz = F.outline(shape, xc)
        ax.plot(ox, oz, color=F.INK, lw=0.9)
        ax.text(np.mean(ox[:2]), min(oz) - 18, name, ha="center", va="bottom", color=F.INK, fontsize=8.5,
                fontweight="bold")
    ax.set_ylim(620, 0)
    ax.set_xlim(-150, 1650)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Uncoupled L1–L2 models: sin θ between ∇Δρ and ∇χ (grey: only the density varies)", loc="left")
    cb = fig.colorbar(im, ax=ax, location="right", shrink=0.85)
    cb.set_label("sin θ (0 = parallel)", color=F.INK2)
    cb.outline.set_visible(False)
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(sins, bins=np.linspace(0, 1, 21), color=F.SERIES, edgecolor="white", linewidth=1)
    ax2.set_xlabel("sin θ where both models vary")
    ax2.set_ylabel("cells")
    ax2.grid(axis="y", color=F.GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"median {stats['median_sin']:.2f}; {stats['cells_only_density_varies']} cells vary in Δρ only",
                  loc="left")
    path = F.OUT / f"{shape}_6_crossgradient.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, stats


def shrinkage_figure():
    """The one step in which the two runs differ: how a cell's (Δρ, χ) is shrunk towards zero.
    Axes in the solver's scaled units, threshold 1 (an illustration, not data)."""
    from matplotlib.patches import Rectangle, Wedge

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    grey, zero = "#e3e2dc", "#cfcdc4"
    P, Q = np.array([1.8, 0.6]), np.array([1.8, 0.0])        # a cell of B, a cell of C
    for ax, coupled in zip(axes, (False, True)):
        if coupled:
            ax.add_patch(Wedge((0, 0), 1.0, 0, 90, facecolor=zero, edgecolor="none"))
            ax.text(0.34, 0.34, "both\nset to 0", ha="center", va="center", fontsize=8.5, color=F.INK)
            p_new = P * (1 - 1 / np.linalg.norm(P))           # the pair scaled down together
            title = "Coupled (group lasso): the pair shrunk together"
            rule = "cell set to 0 only if √(Δρ² + χ²) < threshold;\notherwise both kept, scaled by the same factor"
        else:
            ax.add_patch(Rectangle((0, 0), 1.0, 2.2, facecolor=grey, edgecolor="none"))
            ax.add_patch(Rectangle((0, 0), 2.2, 1.0, facecolor=grey, edgecolor="none"))
            ax.add_patch(Rectangle((0, 0), 1.0, 1.0, facecolor=zero, edgecolor="none"))
            ax.text(0.5, 1.65, "Δρ set\nto 0", ha="center", va="center", fontsize=8.5, color=F.INK)
            ax.text(1.6, 0.9, "χ set to 0", ha="center", va="center", fontsize=8.5, color=F.INK)
            ax.text(0.5, 0.5, "both\nset to 0", ha="center", va="center", fontsize=8.5, color=F.INK)
            p_new = np.array([P[0] - 1, 0.0])                 # each value shrunk by the threshold on its own
            title = "Uncoupled (ordinary L1): each value shrunk on its own"
            rule = "Δρ set to 0 if |Δρ| < threshold; χ set to 0 if |χ| < threshold;\neach independently of the other"
        q_new = np.array([Q[0] - 1, 0.0])
        ax.annotate("", xy=p_new, xytext=P, arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=1.4))
        ax.annotate("", xy=q_new + [0, 0.012], xytext=Q + [0, 0.012],
                    arrowprops=dict(arrowstyle="-|>", color=F.INK2, lw=1.0, ls="--"))
        ax.plot(*P, "o", ms=8, color="#eb6834", mec="white", mew=1.5, zorder=5)
        ax.plot(*p_new, "o", ms=8, color="#eb6834", mec=F.INK, mew=1.2, zorder=5, fillstyle="none")
        ax.text(P[0] + 0.06, P[1] + 0.08, "cell of B:\nclear Δρ, weak χ", fontsize=8.5, color=F.INK)
        ax.text(p_new[0] + 0.05, p_new[1] + 0.07, f"after: χ = {p_new[1]:.2f}", fontsize=8.5, color="#9c3c15",
                fontweight="bold")
        if not coupled:
            ax.text(p_new[0] + 0.05, p_new[1] + 0.2, "B's cell now looks like C's", fontsize=8, color=F.INK2)
        ax.plot(*Q, "s", ms=7, color="#256abf", mec="white", mew=1.5, zorder=5)
        ax.text(Q[0] + 0.06, Q[1] + 0.07, "cell of C:\nno χ → stays 0", fontsize=8.5, color=F.INK)
        ax.set_xlim(0, 2.35)
        ax.set_ylim(0, 2.2)
        ax.set_aspect("equal")
        ax.set_xlabel("density contrast Δρ (scaled)")
        ax.set_ylabel("susceptibility χ (scaled)")
        ax.set_title(title, loc="left")
        ax.text(0.02, 2.13, rule, va="top", fontsize=8, color=F.INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("One step of every iteration: shrinking a cell's trial values towards zero (threshold = 1)",
                 x=0.01, ha="left", fontsize=10, color=F.INK)
    path = F.OUT / "shrinkage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(shapes):
    print(shrinkage_figure())
    for shape in shapes:
        folder = HERE / shape
        table = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        runs = {k: load_result(folder / f"result_{k}.zip") for k, _ in ROWS}
        p1 = why_figure(shape, runs, table)
        p2, stats = crossgradient_figure(shape, runs["none"])
        (folder / "why.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print(p1, p2, stats)


if __name__ == "__main__":
    main(sys.argv[1:] or ["blocks", "dipping"])
