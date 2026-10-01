"""How the models of the three Karnataka studies sit against the published geology: the numbers
behind the validation of the mineral assessment (build_report.py).

The published localities and the schematic geological map are those of
karnataka_inputs/shared/literature.py.  The model maps are the vertically integrated models of
the 1 km core columns: the joint runs with the rock-sample constraints and without them
(karnataka_joint/data/ec2_runs_bounds, ec2_runs_fixed), the single magnetic runs
(karnataka_magnetic/data/ec2_runs: beta1, beta1_mvi) and the gravity run with terrain
(karnataka_gravity_terrain/data/ec2_runs/as1_beta1).  The depths of the gravity and magnetic
reports come from their numbers.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent
MAGNETIC = OUTPUT / "karnataka_magnetic" / "data" / "ec2_runs"
GRAVITY_TERRAIN = OUTPUT / "karnataka_gravity_terrain" / "data" / "ec2_runs" / "as1_beta1"
CONSTRAINED = OUTPUT / "karnataka_joint" / "data" / "ec2_runs_bounds"
UNCONSTRAINED = OUTPUT / "karnataka_joint" / "data" / "ec2_runs_fixed"
DENSE, MAGNETIC_STRONG = 0.75, 0.5       # g/cc·km and SI·km: a dense, a strongly magnetic column


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
