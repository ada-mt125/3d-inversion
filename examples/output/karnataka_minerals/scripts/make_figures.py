"""Figures and numbers of the mineral synthesis of the Sandur area.

    py examples/output/karnataka_minerals/scripts/make_figures.py

Reads the results of the three Karnataka studies (no new inversion): the joint run with the
rock-sample constraints (karnataka_joint/data/ec2_runs_bounds/none_rho35_gb05), the magnetic
runs along the present field and as a magnetization vector (karnataka_magnetic/data/ec2_runs:
beta1, beta1_mvi), and the published localities and schematic map of
karnataka_inputs/shared/literature.py.  Writes figures/*.png and figures/numbers.json.

The prospectivity screens are knowledge-driven and work on the 1 km columns of the models:
- iron-formation horizons: integrated susceptibility ≥ 0.5 SI·km within 1.5 km of a schematic
  ridge or inside a schist belt; a horizon more than 3 km from every iron mine is "not mined";
- the remanent zone: integrated MVI amplitude ≥ 0.5 SI·km where the susceptibility model
  underfits the data by more than 150 nT (within 2 km of such a station);
- dense, magnetite-poor ground (for gold in sulphide-facies units): within 2 km of a dense column
  (integrated density ≥ 0.75 g/cc·km) and within 3 km of an iron-formation horizon, but itself
  weakly magnetic (< 0.2 SI·km).  It is reported, not mapped: it covers most of the belt and
  does not single out targets at 1 km.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent
sys.path.insert(0, str(OUTPUT.parents[1]))
sys.path.insert(0, str(OUTPUT / "karnataka_inputs" / "shared"))
import literature as L  # noqa: E402
from kmodel import Grid, gridded, load  # noqa: E402

FIGS = ROOT / "figures"
JOINT = OUTPUT / "karnataka_joint" / "data" / "ec2_runs_bounds" / "none_rho35_gb05"
MAGNETIC = OUTPUT / "karnataka_magnetic" / "data" / "ec2_runs"
DENSE, MAGNETIC_STRONG, WEAK = 0.75, 0.5, 0.2      # g/cc·km, SI·km, SI·km
RIDGE_KM, MINE_KM, GOLD_DENSE_KM, GOLD_BIF_KM, UNDERFIT_NT = 1.5, 3.0, 2.0, 3.0, 150.0
SECTION_N = 1664.5        # northing of the section (km): through the main high, Donimalai and Kumaraswamy


def integrated(grid):
    """Cell centres (km) of the 1 km core columns and the integrated model [northing, easting]."""
    total, _ = grid.integrated()
    return grid.mesh.cell_centers_x[grid.sx] / 1e3, grid.mesh.cell_centers_y[grid.sy] / 1e3, total.T


def on_grid(xc, yc, field_x, field_y, field):
    """A boolean field of the schematic (450 m) sampled at the column centres."""
    ix = np.abs(field_x[None, :] - xc[:, None]).argmin(axis=1)
    iy = np.abs(field_y[None, :] - yc[:, None]).argmin(axis=1)
    return field[np.ix_(iy, ix)]


def distance_km(mask, cell_km=1.0):
    return ndi.distance_transform_edt(~mask) * cell_km


def town_near(x, y):
    name, tx, ty = min(L.TOWNS, key=lambda t: (t[1] - x) ** 2 + (t[2] - y) ** 2)
    return name, float(np.hypot(tx - x, ty - y))


def screens():
    """The three screens and the maps they rest on."""
    r = load(JOINT)
    rho = Grid(r, r["_models"]["gravity"])
    chi = Grid(load(MAGNETIC / "beta1"))
    mvi_run, ind_run = load(MAGNETIC / "beta1_mvi"), load(MAGNETIC / "beta1")
    mvi = Grid(mvi_run)
    xc, yc, dens = integrated(rho)
    _, _, sus = integrated(chi)
    _, _, amp = integrated(mvi)
    sx, sy, _, ridge, belt, _ = L.schematic()
    near_ridge = on_grid(xc, yc, sx, sy, ndi.distance_transform_edt(~ridge) * float(np.median(np.diff(sx))) <= RIDGE_KM)
    in_belt = on_grid(xc, yc, sx, sy, belt)
    X, Y = np.meshgrid(xc, yc)
    mine_d = np.min([np.hypot(X - e, Y - n) for _, e, n, *_ in L.MINES], axis=0)
    bif = (sus >= MAGNETIC_STRONG) & (near_ridge | in_belt)
    unmined = bif & (mine_d > MINE_KM)
    # the stations the susceptibility model underfits, on the column grid
    d = ind_run["_data"]
    bad = (d["observed"] - d["predicted"]) > UNDERFIT_NT
    under = np.zeros_like(bif)
    for e, n in d["locations"][bad, :2] / 1e3:
        under |= np.hypot(X - e, Y - n) <= 2.0
    remanent = (amp >= MAGNETIC_STRONG) & under
    # its main part: within 10 km of Kumaraswamy (the rest are patches on the north-eastern belt)
    remanent_main = remanent & (np.hypot(X - L.MINES[0][1], Y - L.MINES[0][2]) <= 10.0)
    gold = (distance_km(dens >= DENSE) <= GOLD_DENSE_KM) & (distance_km(bif) <= GOLD_BIF_KM) & (sus < WEAK)
    _, _, _, _, _, belts = L.schematic()
    ne_belt = on_grid(xc, yc, sx, sy, belts == 2)
    return dict(xc=xc, yc=yc, dens=dens, sus=sus, amp=amp, bif=bif, unmined=unmined, remanent=remanent, gold=gold,
                remanent_main=remanent_main, ne_belt=ne_belt,
                mine_d=mine_d, rho=rho, chi=chi, ind=ind_run, bad=bad, in_belt=in_belt)


def segments(S):
    """The iron-formation horizons more than 3 km from every mine, as connected segments."""
    lab, n = ndi.label(S["unmined"], structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 3:              # at least 3 km² of columns
            continue
        Y, X = np.meshgrid(S["yc"], S["xc"], indexing="ij")
        pts = np.c_[X[m], Y[m]]
        c = pts - pts.mean(axis=0)
        length = float(np.ptp(c @ np.linalg.eigh(np.cov(c.T))[1][:, 1]) + 1.0) if m.sum() > 2 else 1.0
        e, nn = pts.mean(axis=0)
        town, dist = town_near(e, nn)
        out.append({"easting": float(e), "northing": float(nn), "cells": int(m.sum()), "length_km": length,
                    "mean_susceptibility": float(S["sus"][m].mean()), "max_susceptibility": float(S["sus"][m].max()),
                    "in_belt": float(S["in_belt"][m].mean()), "dense_share": float((S["dens"][m] >= DENSE).mean()),
                    "nearest_mine_km": float(S["mine_d"][m].min()), "town": town, "town_km": dist,
                    "edge": bool((e < L.AOI_KM[0] + 5) or (e > L.AOI_KM[1] - 5) or (nn < L.AOI_KM[2] + 5)
                                 or (nn > L.AOI_KM[3] - 5))})
    return sorted(out, key=lambda s: -s["cells"])


def at(S, mask, x, y):
    i, j = int(np.argmin(np.abs(S["yc"] - y))), int(np.argmin(np.abs(S["xc"] - x)))
    return bool(mask[i, j])


def mincheri(S):
    """The models under the Mincheri block (the part inside the area), against the rest of the area outside the
    schist belts, and the depth of its dense rock in the joint model."""
    w, e, so, no = L.MINCHERI
    X, Y = np.meshgrid(S["xc"], S["yc"])
    box = (X >= w) & (X <= e) & (Y >= so) & (Y <= no)
    outside = ~S["in_belt"] & ~box
    g = S["rho"]
    cols = (g.mesh.cell_centers_x >= w * 1e3) & (g.mesh.cell_centers_x <= e * 1e3)
    rows = (g.mesh.cell_centers_y >= so * 1e3) & (g.mesh.cell_centers_y <= no * 1e3)
    sel = cols[:, None, None] & rows[None, :, None] & g.active & (g.m > 0.05)
    a = (g.m * g.vol)[sel]
    return {"km2": int(box.sum()), "mean_density": float(S["dens"][box].mean()), "max_density": float(S["dens"][box].max()),
            "dense_share": float((S["dens"][box] >= DENSE).mean()),
            "outside_dense_share": float((S["dens"][outside] >= DENSE).mean()),
            "mean_susceptibility": float(S["sus"][box].mean()), "max_susceptibility": float(S["sus"][box].max()),
            "dense_centroid_km": float((a @ g.depth[sel]) / a.sum() / 1e3) if a.size else None}


def numbers(S, segs):
    area = S["bif"].size
    locs = {}
    for name, x, y, kind in L.LOCALITIES:
        i, j = int(np.argmin(np.abs(S["yc"] - y))), int(np.argmin(np.abs(S["xc"] - x)))
        near = lambda m, km: bool(distance_km(m)[i, j] <= km)   # noqa: E731
        locs[name] = {"kind": kind, "bif_within_1_5km": near(S["bif"], 1.5), "gold_screen": at(S, S["gold"], x, y),
                      "gold_within_1km": near(S["gold"], 1.0), "remanent_within_2km": near(S["remanent"], 2.0),
                      "density": float(S["dens"][i, j]), "susceptibility": float(S["sus"][i, j])}
    rem = S["remanent_main"]
    Y, X = np.meshgrid(S["yc"], S["xc"], indexing="ij")
    b = L.KUMARASWAMY_BLOCKS["B"]
    ne = S["ne_belt"]
    ne_mag = (S["sus"] >= MAGNETIC_STRONG) & ne
    return {
        "thresholds": {"dense": DENSE, "magnetic": MAGNETIC_STRONG, "weak": WEAK, "ridge_km": RIDGE_KM,
                       "mine_km": MINE_KM, "gold_dense_km": GOLD_DENSE_KM, "gold_bif_km": GOLD_BIF_KM},
        "bif_km2": int(S["bif"].sum()), "bif_share": float(S["bif"].mean()),
        "unmined_km2": int(S["unmined"].sum()), "gold_km2": int(S["gold"].sum()), "gold_share": float(S["gold"].mean()),
        "remanent_km2": int(rem.sum()), "remanent_all_km2": int(S["remanent"].sum()),
        "belt_km2": int(S["in_belt"].sum()), "gold_in_belt_share": float(S["gold"][S["in_belt"]].mean()),
        "ne_belt_km2": int(ne.sum()), "ne_belt_magnetic_km2": int(ne_mag.sum()),
        "ne_belt_dense_share": float((S["dens"][ne] >= DENSE).mean()) if ne.any() else 0.0,
        "ne_belt_mean_density": float(S["dens"][ne].mean()) if ne.any() else 0.0,
        "mincheri": mincheri(S),
        "remanent_south_km": float(b[2] - Y[rem].min()) if rem.any() else 0.0,
        "remanent_extent": [float(X[rem].min()), float(X[rem].max()), float(Y[rem].min()), float(Y[rem].max())]
        if rem.any() else None,
        "segments": segs, "localities": locs, "schematic": L.belt_stats(), "area_km2": int(area),
    }


def figures(S, segs, N):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
    FIGS.mkdir(parents=True, exist_ok=True)

    # ── the schematic geology ──
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    handles = L.draw_schematic(ax)
    L.plot_localities(ax)
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92)
    ax.set_xlabel("Easting (km, UTM 43N)")
    ax.set_ylabel("Northing (km)")
    fig.savefig(FIGS / "geology.png")
    plt.close(fig)

    # ── the models: density, susceptibility, magnetization amplitude ──
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), layout="constrained")
    for ax, v, cmap, lo, hi, unit, title in (
            (axs[0], S["dens"], "RdBu_r", -1.5, 1.5, "g/cc·km", "(a) Density contrast, joint run with the sample bounds"),
            (axs[1], S["sus"], "Oranges", 0, 1.0, "SI·km", "(b) Susceptibility along the present field (β = 1)"),
            (axs[2], S["amp"], "Purples", 0, 1.0, "SI·km", "(c) Magnetization-vector amplitude (MVI, β = 1)")):
        im = ax.pcolormesh(S["xc"], S["yc"], v, cmap=cmap, vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        L.outline(ax, color="k", lw=1.0)
        L.plot_localities(ax, labels=False)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Easting (km, UTM 43N)")
        fig.colorbar(im, ax=ax, shrink=0.75, label=f"integrated over depth ({unit})", extend="both" if lo < 0 else "max")
    axs[0].set_ylabel("Northing (km)")
    fig.savefig(FIGS / "models.png")
    plt.close(fig)

    # ── an E–W section through the belt, with the interpretation ──
    from matplotlib.colors import PowerNorm
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    r = load(JOINT)
    chi_c = Grid(r, r["_models"]["magnetics"])
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, layout="constrained")
    northing = SECTION_N
    for ax, g, style, unit in ((axs[0], S["rho"], dict(cmap="RdBu_r", vmin=-0.35, vmax=0.35), "density contrast (g/cc)"),
                               (axs[1], chi_c, dict(cmap="Oranges", norm=PowerNorm(0.5, vmin=0, vmax=1.0)),
                                "susceptibility (SI)")):
        x, z, v, ground, northing = g.section(SECTION_N * 1e3)
        northing /= 1e3
        im = ax.pcolormesh(x, z, v, rasterized=True, **style)
        ax.plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
        ax.set_ylim(-8.5, 1.6)
        ax.set_ylabel("Elevation (km)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
        for name, e, n, *_ in L.MINES:
            if abs(n - SECTION_N) <= 3.5:
                ax.annotate(name, (e, np.interp(e, (x[:-1] + x[1:]) / 2, ground)), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.7),
                            path_effects=halo)
    axs[0].set_title(f"(a) Density contrast, joint run with the sample bounds · northing {northing:.1f} km", loc="left",
                     fontsize=9.5)
    axs[1].set_title("(b) Susceptibility, the same run", loc="left", fontsize=9.5)
    for text, xx, zz in (("metavolcanic core", 668.5, -2.6), ("granite and gneiss", 650.0, -3.5),
                         ("granite and gneiss", 692.0, -3.5)):
        axs[0].text(xx, zz, text, ha="center", fontsize=8, style="italic", path_effects=halo)
    axs[1].text(671.5, -3.9, "iron-formation sheets on both limbs, steep", ha="center", fontsize=8, style="italic",
                path_effects=halo)
    axs[1].set_xlabel("Easting (km, UTM 43N)")
    fig.savefig(FIGS / "section.png")
    plt.close(fig)

    # ── the prospectivity screens ──
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(9, 8))
    sx, sy, _, ridge, belt, _ = L.schematic()
    ax.contourf(sx, sy, belt.astype(float), levels=[0.5, 1.5], colors=["#e9f2e4"])
    classes = np.zeros_like(S["dens"])
    classes[S["bif"]] = 2
    classes[S["unmined"]] = 3
    classes[S["remanent_main"]] = 4
    cmap = ListedColormap(["none", "#f6c75e", "#8c8c8c", "#c0392b", "#6c3483"])
    ax.pcolormesh(S["xc"], S["yc"], np.ma.masked_equal(classes, 0), cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest",
                  rasterized=True)
    L.outline(ax, color="#2d4f2a", lw=0.9)
    L.plot_localities(ax)
    for i, s in enumerate(segs[:8], 1):
        ax.annotate(f"F{i}", (s["easting"], s["northing"]), fontsize=8, fontweight="bold", color="#7b241c",
                    xytext=(6, 6), textcoords="offset points", path_effects=halo)
    ax.legend(handles=[Patch(fc="#8c8c8c", label=f"iron-formation horizon within {MINE_KM:g} km of a located mine"),
                       Patch(fc="#c0392b", label=f"iron-formation horizon farther from them (F1, F2, …)"),
                       Patch(fc="#6c3483", label="remanent zone south of Kumaraswamy (MVI)"),
                       Patch(fc="#e9f2e4", ec="#2d4f2a", label="schist belts (schematic)")],
              loc="lower right", fontsize=7.5, framealpha=0.93)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 43N)")
    ax.set_ylabel("Northing (km)")
    fig.savefig(FIGS / "prospectivity.png")
    plt.close(fig)


def main():
    S = screens()
    segs = segments(S)
    N = numbers(S, segs)
    figures(S, segs, N)
    (FIGS / "numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    print("figures and numbers in", FIGS)


if __name__ == "__main__":
    main()
