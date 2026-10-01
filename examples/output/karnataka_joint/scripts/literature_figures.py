"""Figures and numbers of the assessment against the published geology (Section 7 of the second
joint report).  Called by make_figures.py --set v2.

The published localities and the schematic geological map are those of
karnataka_inputs/shared/literature.py.  The model maps are the vertically integrated models of
the 1 km core columns: the joint runs with the rock-sample constraints (data/ec2_runs_bounds),
the single magnetic runs (karnataka_magnetic/data/ec2_runs: beta1, beta1_mvi) and the gravity
run with terrain (karnataka_gravity_terrain/data/ec2_runs/as1_beta1).  The depths of the other
two reports come from their numbers.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent
MAGNETIC = OUTPUT / "karnataka_magnetic" / "data" / "ec2_runs"
GRAVITY_TERRAIN = OUTPUT / "karnataka_gravity_terrain" / "data" / "ec2_runs" / "as1_beta1"
CONSTRAINED = ROOT / "data" / "ec2_runs_bounds"
UNCONSTRAINED = ROOT / "data" / "ec2_runs_fixed"
DENSE, MAGNETIC_STRONG = 0.75, 0.5       # g/cc·km and SI·km: a dense, a strongly magnetic column
UNDERFIT_NT = 150.0


def _maps(grid):
    """Cell centres (km) of the core columns and the integrated model, indexed [northing, easting]."""
    total, _ = grid.integrated()
    xc = grid.mesh.cell_centers_x[grid.sx] / 1e3
    yc = grid.mesh.cell_centers_y[grid.sy] / 1e3
    return xc, yc, total.T


def _belt_shape(xc, yc, value, threshold):
    """Length, largest width (km) and strike (degrees) of the largest connected set of columns
    with value ≥ threshold."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(value >= threshold)
    if n == 0:
        return None
    k = 1 + int(np.argmax(ndi.sum(np.ones_like(value), lab, range(1, n + 1))))
    Y, X = np.meshgrid(yc, xc, indexing="ij")
    pts = np.c_[X[lab == k], Y[lab == k]]
    c = pts - pts.mean(axis=0)
    _, v = np.linalg.eigh(np.cov(c.T))
    s, t = c @ v[:, 1], c @ v[:, 0]
    widths = [np.ptp(t[(s >= a) & (s < a + 1.0)]) + 1.0 for a in np.arange(s.min(), s.max() + 1.0)
              if ((s >= a) & (s < a + 1.0)).sum() > 0]
    return {"length_km": float(np.ptp(s) + 1.0), "width_km": float(max(widths)),
            "strike_deg": float((np.degrees(np.arctan2(v[0, 1], v[1, 1])) + 360) % 180),
            "area_km2": float(len(pts) * np.median(np.diff(xc)) ** 2)}


def _on_schematic(xc, yc, value, threshold):
    """The share of the columns with value ≥ threshold inside the schematic belt outline and
    within 1 km of a schematic ridge, against the share of the area (5 km edge band left out)."""
    from scipy import ndimage as ndi
    import literature as L
    sx, sy, _, ridge, belt, _ = L.schematic()
    # the schematic grids (450 m) sampled at the model's column centres
    ix = np.abs(sx[None, :] - xc[:, None]).argmin(axis=1)     # nearest (the DEM rows run north to south)
    iy = np.abs(sy[None, :] - yc[:, None]).argmin(axis=1)
    near_ridge = ndi.distance_transform_edt(~ridge) * float(np.median(np.diff(sx))) <= 1.0
    inside, near = belt[np.ix_(iy, ix)], near_ridge[np.ix_(iy, ix)]
    # a 5 km edge band left out, as in the magnetic report
    edge = 5.0
    keep = ((xc[None, :] >= xc.min() + edge) & (xc[None, :] <= xc.max() - edge)
            & (yc[:, None] >= yc.min() + edge) & (yc[:, None] <= yc.max() - edge))
    strong = (value >= threshold) & keep
    return {"in_outline": float(inside[strong].mean()), "area_in_outline": float(inside[keep].mean()),
            "near_ridge": float(near[strong].mean()), "area_near_ridge": float(near[keep].mean())}


def numbers():
    import literature as L
    from kmodel import Grid, load
    out = {"thresholds": {"dense": DENSE, "magnetic": MAGNETIC_STRONG}, "schematic": L.belt_stats()}
    # the joint result keeps one model per method
    r = load(CONSTRAINED / "none_rho35_gb05")
    g_rho, g_chi = Grid(r, r["_models"]["gravity"]), Grid(r, r["_models"]["magnetics"])
    r0 = load(UNCONSTRAINED / "none")
    g_rho0 = Grid(r0, r0["_models"]["gravity"])
    single = {"susceptibility": Grid(load(MAGNETIC / "beta1")), "mvi": Grid(load(MAGNETIC / "beta1_mvi")),
              "density": Grid(load(GRAVITY_TERRAIN))}
    maps = {"density_constrained": (_maps(g_rho), DENSE), "density_unconstrained": (_maps(g_rho0), DENSE),
            "susceptibility_constrained": (_maps(g_chi), MAGNETIC_STRONG),
            "susceptibility_single": (_maps(single["susceptibility"]), MAGNETIC_STRONG),
            "density_single": (_maps(single["density"]), DENSE), "mvi_single": (_maps(single["mvi"]), MAGNETIC_STRONG)}
    out["shape"] = {k: _belt_shape(*m, t) for k, (m, t) in maps.items() if k.startswith("density")}
    out["schematic_match"] = {k: _on_schematic(*m, t) for k, (m, t) in maps.items()}
    out["distances"] = {k: L.distance_table(*m, t) for k, (m, t) in maps.items()}
    # the depths of the other two reports (their numbers.json)
    mag = json.loads((OUTPUT / "karnataka_magnetic" / "figures" / "numbers.json").read_text())
    grav = json.loads((OUTPUT / "karnataka_gravity" / "figures" / "numbers.json").read_text())
    cen = [mag["full"][k]["box_sandur"]["centroid_km"] for k in ("beta0.5", "beta1", "l1l2_irls") if k in mag["full"]]
    out["magnetic_centroid_km"] = [min(cen), max(cen)]
    out["gravity_base_km"] = {k: grav["full"][k]["main"]["bottom_km"] for k in ("as1_beta0.5", "as1_beta1", "as1_beta1.5")
                              if k in grav["full"]}
    out["cell_thickness_m"] = float(np.min(g_rho.dz[g_rho.zc >= g_rho.z_core_base]))
    return out


def figures(figs, nums, C):
    """lit_mvi: the localities on the magnetic residuals and the MVI amplitude; lit_depths: the
    published depths against the scales of the inversions; lit_geology: the schematic map
    beside the constrained models."""
    import matplotlib.pyplot as plt
    import literature as L
    from kmodel import Grid, gridded, load
    a, b = load(MAGNETIC / "beta1"), load(MAGNETIC / "beta1_mvi")
    bad = (a["_data"]["observed"] - a["_data"]["predicted"]) > UNDERFIT_NT

    # ── the localities on the magnetic residuals and on the MVI amplitude ──
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    d = a["_data"]
    x, y, g = gridded(d["locations"], d["observed"] - d["predicted"])
    im = axs[0].pcolormesh(x, y, g, cmap="RdBu_r", vmin=-150, vmax=150, shading="nearest", rasterized=True)
    axs[0].plot(d["locations"][bad, 0] / 1e3, d["locations"][bad, 1] / 1e3, "k.", ms=2.5)
    axs[0].set_title("Magnetic residual, susceptibility inversion (β = 1)", fontsize=9.5)
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="observed − predicted (nT)", extend="both")
    xc, yc, amp = _maps(Grid(b))
    im = axs[1].pcolormesh(xc, yc, amp, cmap="Oranges", vmin=0, vmax=1.0, shading="nearest", rasterized=True)
    axs[1].plot(d["locations"][bad, 0] / 1e3, d["locations"][bad, 1] / 1e3, "k.", ms=2.5)
    axs[1].set_title("MVI amplitude integrated with depth", fontsize=9.5)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="|m| × thickness (SI·km)", extend="max")
    for ax in axs:
        L.plot_localities(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km, UTM 43N)")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(figs / "lit_mvi.png")
    plt.close(fig)

    # ── depths: published and mined against the scales of the inversions ──
    k = C["full_key"]
    main = C["full"][k]["none"]["main"]
    m0, m1 = nums["magnetic_centroid_km"]
    rows = [  # label, from m, to m (or a point), published?, note
        ("Kumaraswamy ore bodies, mean drilled depth", 10, 70, True, "IBM inspection report"),
        ("Donimalai supergene enrichment to ore grade", 10, 170, True, "ROM BIF note"),
        ("BIF bands, ground magnetics (tops)", 70, 130, True, "Satish Kumar et al. 2018"),
        ("Joint inversion cell thickness", 10, nums["cell_thickness_m"], False, "one cell"),
        ("Magnetic rock of the belt, centroid range", m0 * 1e3, m1 * 1e3, False, "β 0.5 to 1, L1–L2"),
        ("Dense body, half-maximum (constrained)", main["top_km"] * 1e3, main["bottom_km"] * 1e3, False,
         "sample bounds, β 0.5"),
        ("Dense body, base (constrained)", main["dense_bottom_km"] * 1e3, None, False, "> +0.05 g/cc"),
        ("Schist belt basin depth (joint grav–mag)", 6000, None, True, "Maurya et al. 2023"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (label, lo, hi, pub, note) in enumerate(rows[::-1]):
        col = "#e8743b" if pub else "#2a78d6"
        if hi is None:
            ax.plot(lo, i, "D", color=col, ms=8)
            ax.text(lo * 1.12, i, note, va="center", fontsize=8.5, color="#444")
        else:
            ax.barh(i, hi - lo, left=lo, height=0.45, color=col)
            ax.text(hi * 1.12, i, note, va="center", fontsize=8.5, color="#444")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(10, 60000)
    ax.set_xticks([10, 100, 1000, 10000], ["10 m", "100 m", "1 km", "10 km"])
    ax.set_xlabel("Depth below the ground (log scale)")
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e8743b", label="Published / mining records"),
                       Patch(color="#2a78d6", label="This study (inversion)")], loc="upper right", frameon=False)
    fig.savefig(figs / "lit_depths.png")
    plt.close(fig)

    # ── the schematic geological map beside the constrained models ──
    r = load(CONSTRAINED / "none_rho35_gb05")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained")
    handles = L.draw_schematic(axs[0])
    L.plot_localities(axs[0])
    axs[0].legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    axs[0].set_title("(a) Schematic geology (from the DEM; not a geological map)", fontsize=9.5)
    for ax, model, cmap, vmin, vmax, unit, title in (
            (axs[1], r["_models"]["gravity"], "RdBu_r", -1.5, 1.5, "integrated density (g/cc·km)",
             "(b) Density contrast, constrained (integrated)"),
            (axs[2], r["_models"]["magnetics"], "Oranges", 0, 1.0, "integrated susceptibility (SI·km)",
             "(c) Susceptibility, constrained run (integrated)")):
        xc, yc, v = _maps(Grid(r, model))
        im = ax.pcolormesh(xc, yc, v, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
        L.outline(ax, color="k", lw=1.0)
        L.plot_localities(ax, labels=False)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax, shrink=0.75, label=unit, extend="both" if vmin < 0 else "max")
    for ax in axs:
        ax.set_xlabel("Easting (km, UTM 43N)")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(figs / "lit_geology.png")
    plt.close(fig)


def literature(figs, C):
    nums = numbers()
    figures(figs, nums, C)
    return nums
