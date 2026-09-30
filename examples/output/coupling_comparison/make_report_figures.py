"""Figures of the coupling comparison: the synthetic model, every coupling's sections
(susceptibility and density, through the bodies' centre line), and the data fit.

    py examples/output/coupling_comparison/make_report_figures.py [blocks] [dipping]

Reads ./<shape>/result_*.zip and comparison.json (run_coupling_comparison.py), writes
./report/<shape>_*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from geoinv3d.methods.coupling import coupling_label   # noqa: E402
from geoinv3d.viz.result_workflow import load_result, result_mesh   # noqa: E402

OUT = HERE / "report"
OUT.mkdir(exist_ok=True)
KEYS = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso"]
SHORT = {"none": "No coupling", "cross_gradient": "Cross-gradient", "joint_total_variation": "Joint total variation",
         "linear_correspondence": "Linear correspondence", "pgi": "PGI (rock units)",
         "group_lasso": "Group lasso (Utsugi 2025)"}
BODIES = [("A", "high χ", 300.0, 0.05), ("B", "low χ", 750.0, 0.01), ("C", "no χ", 1200.0, 0.0)]
RHO, Y0 = 0.3, 600.0

# one-hue sequential ramps (light = near zero): density blue, susceptibility orange
BLUE = ["#f3f7fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fdf4ee", "#fbd9c6", "#f6b594", "#f08f63", "#eb6834", "#c9501f", "#9c3c15", "#6b280c"]
CM_RHO = LinearSegmentedColormap.from_list("rho", BLUE)
CM_CHI = LinearSegmentedColormap.from_list("chi", ORANGE)
CM_RES = LinearSegmentedColormap.from_list("res", ["#184f95", "#6da7ec", "#f0efec", "#ec8a86", "#b3262b"])
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
SERIES = "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5,
                     "axes.titlecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans"})


def outline(shape, xc):
    """The body's outline in the section (x, depth), closed."""
    if shape == "blocks":
        return [xc - 100, xc + 100, xc + 100, xc - 100, xc - 100], [100, 100, 300, 300, 100]
    top, bottom, t = 75.0, 375.0, np.tan(np.radians(55.0))
    x0, x1 = xc - 100, xc - 100 + (bottom - top) / t
    return [x0 - 60, x0 + 60, x1 + 60, x1 - 60, x0 - 60], [top, top, bottom, bottom, top]


def section(run, values):
    """The model on the x-z section nearest y = 600 m (the two nearest rows averaged)."""
    mesh = result_mesh(run)
    full = np.zeros(mesh.n_cells)
    act = run.get("_active")
    if act is not None:
        full[np.asarray(act, dtype=bool)] = values
    else:
        full[:] = values
    grid = full.reshape(mesh.shape_cells, order="F")
    cy = mesh.cell_centers_y
    near = np.argsort(np.abs(cy - Y0))[:2]
    w = 1.0 / (np.abs(cy[near] - Y0) + 1e-6)
    sec = np.tensordot(grid[:, near, :], w / w.sum(), axes=([1], [0]))   # (nx, nz)
    cx, cz = mesh.cell_centers_x, mesh.cell_centers_z
    keep_x = (cx > -150) & (cx < 1650)
    keep_z = (cz < 0) & (cz > -650)
    return cx[keep_x], -cz[keep_z], sec[np.ix_(keep_x, keep_z)].T   # x, depth, (nz, nx)


def draw_section(ax, x, depth, sec, cmap, vmax, shape, title, label_bodies=True):
    im = ax.pcolormesh(x, depth, np.clip(sec, 0, None), cmap=cmap, vmin=0, vmax=vmax, shading="nearest",
                       rasterized=True)
    for name, _, xc, _ in BODIES:
        ox, oz = outline(shape, xc)
        ax.plot(ox, oz, color=INK, lw=0.9)
        if label_bodies:
            ax.text(np.mean(ox[:2]), min(oz) - 18, name, ha="center", va="bottom", color=INK, fontsize=8.5,
                    fontweight="bold")
    ax.set_ylim(620, 0)
    ax.set_xlim(-150, 1650)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    return im


def synthetic_figure(shape, runs):
    run = runs["none"]
    mesh = result_mesh(run)
    # the true models on the same section (from the exact outlines, on a fine grid)
    xs = np.linspace(-150, 1650, 721)
    ds = np.linspace(0, 620, 249)
    X, D = np.meshgrid(xs, ds)
    rho, chi = np.zeros_like(X), np.zeros_like(X)
    for name, _, xc, sus in BODIES:
        from matplotlib.path import Path as MPath
        ox, oz = outline(shape, xc)
        inside = MPath(np.column_stack([ox, oz])).contains_points(np.column_stack([X.ravel(), D.ravel()]))
        rho.ravel()[inside] = RHO
        chi.ravel()[inside] = sus
    fig = plt.figure(figsize=(12.5, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax3 = fig.add_subplot(gs[0], projection="3d")
    for name, desc, xc, sus in BODIES:
        ox, oz = outline(shape, xc)
        verts = []
        for y in (Y0 - 150, Y0 + 150):
            verts.append([(ox[k], y, -oz[k]) for k in range(4)])
        for k in range(4):   # the side faces
            k2 = (k + 1) % 4
            verts.append([(ox[k], Y0 - 150, -oz[k]), (ox[k2], Y0 - 150, -oz[k2]),
                          (ox[k2], Y0 + 150, -oz[k2]), (ox[k], Y0 + 150, -oz[k])])
        ax3.add_collection3d(Poly3DCollection(verts, facecolor="#256abf", edgecolor="#0d366b",
                                              linewidths=0.4, alpha=0.85))
        ax3.text(np.mean(ox[:2]), Y0 - 260, 40, f"{name}", color=INK, fontsize=9, fontweight="bold")
    ax3.set_xlim(0, 1500)
    ax3.set_ylim(0, 1200)
    ax3.set_zlim(-620, 0)
    ax3.set_box_aspect((1.5, 1.2, 0.62))
    ax3.set_xlabel("x (m)", labelpad=2)
    ax3.set_ylabel("y (m)", labelpad=2)
    ax3.set_zlabel("z (m)", labelpad=2)
    ax3.view_init(elev=24, azim=-58)
    ax3.set_title("Synthetic model", loc="left")
    ax3.tick_params(labelsize=7, pad=0)
    for i, (arr, cmap, vmax, title, unit) in enumerate((
            (chi, CM_CHI, 0.05, "True susceptibility, section y = 600 m", "SI"),
            (rho, CM_RHO, RHO, "True density contrast, section y = 600 m", "g/cc"))):
        ax = fig.add_subplot(gs[1 + i])
        im = draw_section(ax, xs, ds, arr, cmap, vmax, shape, title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("depth (m)")
        cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.9, aspect=30)
        cb.set_label(unit, color=INK2)
        cb.outline.set_visible(False)
    ax3.text2D(0.0, -0.02, "A 0.3 g/cc, 0.05 SI · B 0.3 g/cc, 0.01 SI · C 0.3 g/cc, 0 SI",
               transform=ax3.transAxes, color=INK2, fontsize=8)
    path = OUT / f"{shape}_1_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def sections_figure(shape, runs, table):
    fig, axes = plt.subplots(len(KEYS), 2, figsize=(11, 1.95 * len(KEYS) + 1.2), sharex=True, sharey=True,
                             layout="constrained")
    for r, key in enumerate(KEYS):
        run = runs[key]
        for c, (model, cmap, vmax) in enumerate((("magnetics", CM_CHI, 0.05), ("gravity", CM_RHO, RHO))):
            x, d, sec = section(run, run["_models"][model])
            e = table[key]["bodies"]
            fmt = (lambda v: f"{v:.4f}") if model == "magnetics" else (lambda v: f"{v:.3f}")
            vals = "   ".join(f"{b} {fmt(e[b]['chi_mean' if model == 'magnetics' else 'rho_mean'])}"
                               for b in ("A", "B", "C"))
            prop = "susceptibility" if model == "magnetics" else "density contrast"
            im = draw_section(axes[r, c], x, d, sec, cmap, vmax, shape, f"{SHORT[key]} · {prop}",
                              label_bodies=True)
            axes[r, c].text(0.99, 0.04, f"body means  {vals}", transform=axes[r, c].transAxes, ha="right",
                            va="bottom", fontsize=7.5, color=INK,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            if c == 0:
                axes[r, c].set_ylabel("depth (m)")
            if r == len(KEYS) - 1:
                axes[r, c].set_xlabel("x (m)")
            if r == len(KEYS) - 1:
                cb = fig.colorbar(im, ax=axes[:, c], location="bottom", shrink=0.9, aspect=40)
                cb.set_label("susceptibility (SI) — true 0.05 / 0.01 / 0" if c == 0
                             else "density contrast (g/cc) — true 0.3", color=INK2)
                cb.outline.set_visible(False)
    fig.suptitle("Sections at y = 600 m on the true model's colour scales (values above them clipped)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    path = OUT / f"{shape}_2_sections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fit_figures(shape, runs, table):
    paths = []
    # (a) profiles along y = 600 m: observed (dots, ±σ) and predicted (line)
    fig, axes = plt.subplots(2, len(KEYS), figsize=(14, 4.9), sharex=True, sharey="row", layout="constrained")
    for c, key in enumerate(KEYS):
        datas = runs[key]["_datas"]
        for r, (label, unit) in enumerate((("gravity", "gz (mGal)"), ("magnetics", "TMI (nT)"))):
            d = datas[label]
            loc = np.asarray(d["locations"])
            row = np.abs(loc[:, 1] - Y0) < 1
            order = np.argsort(loc[row, 0])
            xo = loc[row, 0][order]
            obs, pred = np.asarray(d["observed"])[row][order], np.asarray(d["predicted"])[row][order]
            std = np.asarray(d["std"])[row][order]
            ax = axes[r, c]
            ax.errorbar(xo, obs, yerr=std, fmt="o", ms=3.2, color="#8a8984", ecolor="#c9c8c2", elinewidth=0.8,
                        label="observed ± σ")
            ax.plot(xo, pred, color=SERIES, lw=2, label="predicted")
            n = len(np.asarray(d["observed"]))
            chi2 = table[key]["chi2"][label] / n
            ax.set_title(f"{SHORT[key]}\n{label} χ²/N = {chi2:.2f}" if r == 0 else f"{label} χ²/N = {chi2:.2f}",
                         loc="left", fontsize=8.5)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel(unit)
            if r == 1:
                ax.set_xlabel("x (m)")
            for _, _, xc, _ in BODIES:
                ax.axvline(xc, color=GRID, lw=0.8, zorder=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Data fit along y = 600 m (the bodies' centre line; faint lines mark the bodies' centres)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    path = OUT / f"{shape}_3_fit_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    # (b) normalized residual maps (observed - predicted) / σ
    fig, axes = plt.subplots(2, len(KEYS), figsize=(14, 4.9), sharex=True, sharey=True, layout="constrained")
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    for c, key in enumerate(KEYS):
        for r, label in enumerate(("gravity", "magnetics")):
            d = runs[key]["_datas"][label]
            loc = np.asarray(d["locations"])
            res = (np.asarray(d["observed"]) - np.asarray(d["predicted"])) / np.asarray(d["std"])
            ax = axes[r, c]
            sc = ax.scatter(loc[:, 0], loc[:, 1], c=res, cmap=CM_RES, norm=norm, s=16, marker="s", linewidths=0)
            for _, _, xc, _ in BODIES:
                ax.add_patch(plt.Rectangle((xc - 100, Y0 - 150), 200, 300, fill=False, ec=INK2, lw=0.6))
            ax.set_aspect("equal")
            rms = float(np.sqrt(np.mean(res ** 2)))
            ax.set_title((f"{SHORT[key]}\n" if r == 0 else "") + f"{label}: rms {rms:.2f} σ", loc="left", fontsize=8.5)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("y (m)")
            if r == 1:
                ax.set_xlabel("x (m)")
    cb = fig.colorbar(sc, ax=axes, location="right", shrink=0.85, aspect=30)
    cb.set_label("(observed − predicted) / σ", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Normalized residuals; a good fit is noise: no pattern, within ±2 σ (outlines: the bodies' plan)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    path = OUT / f"{shape}_4_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main(shapes):
    for shape in shapes:
        folder = HERE / shape
        table = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        runs = {k: load_result(folder / f"result_{k}.zip") for k in KEYS}
        for p in [synthetic_figure(shape, runs), sections_figure(shape, runs, table),
                  *fit_figures(shape, runs, table)]:
            print(p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["blocks", "dipping"])
