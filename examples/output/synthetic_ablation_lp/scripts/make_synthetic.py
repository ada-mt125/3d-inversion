"""The second ablation: Lp only, tuned, with a soft reference model and boreholes with a reach.

    py examples/output/synthetic_ablation_lp/scripts/make_synthetic.py

The first study (``../synthetic_ablation``) left every method at one setting, its intrusion was
a 100 m sheet, its prior was pinned by the model builder's default range and its holes reached
no further than their trace.  Here:

    the model     the same ground (seed 2026), field and data as before; the cube as before
                  (300 m, 100–400 m below the ground, 0.5 SI); the intrusion stubbier: 250 m
                  thick, 1.2 km along strike, 100–600 m below the ground, 45° to the east, 1 SI
    the grid      Lp only: 5 norms × 4 depth weightings β × 2 smoothness length scales = 40
                  settings, each run in every group, with up to 200 iterations (100 IRLS)
    A             no prior: the reference 0, bounds 0–3 SI
    B             the bodies drawn ~100 m off, 50 m deeper, 10° steeper, κ ±50 %, as a
                  reference model only (bounds 0–3 SI), at three weights: 1, 10, 100
    C             five holes logging the true model, each reaching 150 m around its trace
                  (``radius_m``: the share falls linearly from 1 at the trace to 0 at 150 m)

Writes ``inputs/``: the DEM, the TMI, the truth, ``spec_B.json`` (weight left to the run),
``boreholes_C.csv``, ``spec_C.json`` and ``runs.json``, the 200 runs' parameters.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import rasterio
from discretize import TensorMesh
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from simpeg import maps
from simpeg.potential_fields import magnetics as mag

OUT = Path(__file__).resolve().parents[1] / "inputs"
X0, Y0 = 600000.0, 1600000.0                      # UTM 43N, the 5 × 5 km survey from here
FIELD = (42085.1, 19.24, -1.32)                   # as at Block 8: IGRF 2020, F (nT), I, D (°)
SEED = 2026

NORMS = {"n0000": [0, 0, 0, 0], "n0111": [0, 1, 1, 1], "n0221": [0, 2, 2, 1], "n0222": [0, 2, 2, 2],
         "n1111": [1, 1, 1, 1]}
BETAS = {"b10": 1.0, "b15": 1.5, "b20": 2.0, "b30": 3.0}
LENGTHS = {"L1": 1.0, "L3": 3.0}                  # alpha_x = alpha_y = alpha_z: SimPEG length scales
B_WEIGHTS = {"w1": 1.0, "w10": 10.0, "w100": 100.0}
HOLE_RADIUS = 150.0


def ground_grid(rng):
    """Random ground: white noise smoothed over 400 m, scaled to ±150 m over the survey."""
    h = 50.0
    xs = np.arange(X0 - 5000, X0 + 10000 + h, h)
    ys = np.arange(Y0 - 5000, Y0 + 10000 + h, h)
    z = gaussian_filter(rng.standard_normal((len(ys), len(xs))), sigma=400.0 / h, mode="reflect")
    inside = (ys >= Y0)[:, None] & (ys <= Y0 + 5000)[:, None] & (xs >= X0) & (xs <= X0 + 5000)
    z = 600.0 + 150.0 * z / np.abs(z[inside]).max()
    return xs, ys, z, h


def bodies(G):
    cube = {"centre": [X0 + 1300.0, Y0 + 2500.0], "edge": 300.0, "top_depth": 100.0, "kappa": 0.5}
    dyke = {"top_x": X0 + 3000.0, "y": [Y0 + 1900.0, Y0 + 3100.0], "top_depth": 100.0, "bottom_depth": 600.0,
            "dip": 45.0, "dip_direction": 90.0, "thickness": 250.0, "kappa": 1.0}
    cz = float(G(np.array([cube["centre"][0]]), np.array([cube["centre"][1]]))[0])
    cube["top_elev"] = cz - cube["top_depth"]
    cube["bottom_elev"] = cube["top_elev"] - cube["edge"]
    return cube, dyke


def true_model(cc, G, cube, dyke):
    """Susceptibility at points (x, y, elevation): the cube (0.5 SI) and the intrusion (1 SI)."""
    half = cube["edge"] / 2
    in_cube = (np.abs(cc[:, 0] - cube["centre"][0]) <= half) & (np.abs(cc[:, 1] - cube["centre"][1]) <= half) \
        & (cc[:, 2] <= cube["top_elev"]) & (cc[:, 2] >= cube["bottom_elev"])
    near = (cc[:, 0] > dyke["top_x"] - 400) & (cc[:, 0] < dyke["top_x"] + 1200) \
        & (cc[:, 1] >= dyke["y"][0]) & (cc[:, 1] <= dyke["y"][1])
    depth = np.full(len(cc), np.nan)
    depth[near] = G(cc[near, 0], cc[near, 1]) - cc[near, 2]
    hw = dyke["thickness"] / np.sin(np.radians(dyke["dip"])) / 2           # half the horizontal width
    xc = dyke["top_x"] + (depth - dyke["top_depth"]) / np.tan(np.radians(dyke["dip"]))
    in_dyke = near & (depth >= dyke["top_depth"]) & (depth <= dyke["bottom_depth"]) & (np.abs(cc[:, 0] - xc) <= hw)
    return np.where(in_dyke, dyke["kappa"], np.where(in_cube, cube["kappa"], 0.0)), in_cube, in_dyke


def prior_B(cube, dyke, weight):
    """The bodies as a geologist might draw them (~100 m off, 50 m deeper, 10° steeper, κ ±50 %),
    as a reference model only: their bounds are the unconstrained ones (0–3 SI)."""
    cx, cy = cube["centre"][0] + 80.0, cube["centre"][1] - 60.0
    e = cube["edge"] / 2
    cube_poly = [[cx - e, cy - e], [cx + e, cy - e], [cx + e, cy + e], [cx - e, cy + e]]
    dip = dyke["dip"] + 10.0
    hw = dyke["thickness"] / np.sin(np.radians(dip)) / 2
    x0 = dyke["top_x"] - 100.0
    y0, y1 = dyke["y"][0] + 100.0, dyke["y"][1] + 100.0
    dyke_poly = [[x0 - hw, y0], [x0 + hw, y0], [x0 + hw, y1], [x0 - hw, y1]]
    rnd = lambda ring: [[round(float(a), 1), round(float(b), 1)] for a, b in ring]   # noqa: E731
    unit = lambda v: {"value": v, "lower": 0.0, "upper": 3.0, "weight": weight}      # noqa: E731
    return {"name": f"B: perturbed reference model, weight {weight:g}", "property": "susceptibility",
            "units": {"cube (prior)": unit(0.75), "intrusion (prior)": unit(0.5)},
            "sources": [{"type": "body", "unit": "cube (prior)", "crs": "job", "polygon": rnd(cube_poly),
                         "top_m": cube["top_depth"] + 50.0, "bottom_m": cube["top_depth"] + cube["edge"] + 50.0},
                        {"type": "body", "unit": "intrusion (prior)", "crs": "job", "polygon": rnd(dyke_poly),
                         "top_m": dyke["top_depth"] + 50.0, "bottom_m": dyke["bottom_depth"] + 50.0,
                         "dip": dip, "dip_direction": 90.0}],
            "unconstrained": {"lower": 0.0, "upper": 3.0}}


HOLES = [  # name, collar x, y, bearing (°), inclination (° below the horizontal), length (m)
    ("H1 cube centre", 601300.0, 1602500.0, 0.0, 90.0, 600.0),
    ("H2 east of the cube", 601500.0, 1602500.0, 0.0, 90.0, 500.0),
    ("H3 intrusion, inclined", 603700.0, 1602500.0, 270.0, 60.0, 800.0),
    ("H4 intrusion, shallow", 603150.0, 1602100.0, 0.0, 90.0, 500.0),
    ("H5 intrusion, deep", 603450.0, 1602900.0, 0.0, 90.0, 800.0),
]
UNIT_OF = {0.0: "hole: background", 0.5: "hole: 0.5 SI", 1.0: "hole: 1 SI"}


def boreholes_C(G, cube, dyke, step=5.0):
    """The true susceptibility logged along five holes, as intervals of one value each; the
    measured values held (±0.005 SI, weight 5) and reaching HOLE_RADIUS around the trace."""
    rows = []
    for name, x, y, az, inc, length in HOLES:
        s = np.arange(0.0, length + step, step)
        a, c = np.radians(az), np.cos(np.radians(inc))
        px, py = x + s * c * np.sin(a), y + s * c * np.cos(a)
        top = float(G(np.array([x]), np.array([y]))[0])
        pz = top - s * np.sin(np.radians(inc))
        k, _, _ = true_model(np.column_stack([px, py, pz]), G, cube, dyke)
        starts = [0] + [i for i in range(1, len(k)) if k[i] != k[i - 1]]
        for j, i0 in enumerate(starts):
            i1 = starts[j + 1] if j + 1 < len(starts) else len(k) - 1
            rows.append({"hole": name, "x": round(px[i0], 2), "y": round(py[i0], 2), "z_top": round(pz[i0], 2),
                         "bearing": az, "inclination": inc, "length_m": round(s[i1] - s[i0], 2),
                         "unit": UNIT_OF[float(k[i0])]})
    spec = {"name": f"C: borehole logs, {HOLE_RADIUS:g} m around the holes", "property": "susceptibility",
            "units": {"hole: background": {"value": 0.0, "lower": 0.0, "upper": 0.005, "weight": 5},
                      "hole: 0.5 SI": {"value": 0.5, "lower": 0.495, "upper": 0.505, "weight": 5},
                      "hole: 1 SI": {"value": 1.0, "lower": 0.995, "upper": 1.005, "weight": 5}},
            "sources": [{"type": "boreholes", "file": "boreholes_C.csv", "crs": "job", "collar": "elevation",
                         "collar_field": "z_top", "azimuth_field": "bearing", "inclination_field": "inclination",
                         "length_field": "length_m", "unit_by": {"field": "unit"}, "radius_m": HOLE_RADIUS}],
            "unconstrained": {"lower": 0.0, "upper": 3.0}}
    return rows, spec


def runs(cube, dyke, spec_c):
    """The 200 runs: the 40 settings in A and C, and in B at each of the three weights."""
    ds = {"type": "magnetic", "method": "magnetic", "files": ["synthetic_tmi_80m.csv"], "noise_pct": 0.02,
          "noise_floor": 5.0, "component": "tmi", "method_kwargs": {"inducing_field": list(FIELD)},
          "decimate_spacing_m": 100.0}
    common = {"method_type": "magnetic", "inversion_mode": "single", "datasets": [ds], "data_file": ds["files"][0],
              "noise_pct": 0.02, "noise_floor": 5.0, "method_kwargs": {"inducing_field": list(FIELD)},
              "topography": {"file": "synthetic_dem.tif"}, "crs": "EPSG:32643", "param_mode": "manual",
              "regularization_type": "sparse", "beta_selection": "auto", "beta0_ratio": 1, "cooling_factor": 2,
              "max_iter": 200, "max_irls_iterations": 100, "use_preconditioner": True,
              "bounds_lower": 0.0, "bounds_upper": 3.0, "alpha_s": 1, "depth_weighting": "depth",
              "core_cell_m": 50.0, "core_cell_z_m": 25.0, "depth_core_m": 2000.0, "pad_distance_m": 3000.0,
              "mesh_type": "octree", "octree_levels": [8, 8, 8]}
    groups = {"A": (None, []), "C": (spec_c, ["boreholes_C.csv"])}
    groups.update({f"B{w}": (prior_B(cube, dyke, v), []) for w, v in B_WEIGHTS.items()})
    out = {}
    for g, (geology, files) in groups.items():
        for (nk, norms), (bk, beta), (lk, ls) in itertools.product(NORMS.items(), BETAS.items(), LENGTHS.items()):
            p = {**copy.deepcopy(common), "norms": norms, "depth_weighting_exponent": beta,
                 "alpha_x": ls, "alpha_y": ls, "alpha_z": ls}
            if geology:
                p["geology"] = geology
            out[f"{g}_{nk}_{bk}_{lk}"] = {"group": g[0], "weight": B_WEIGHTS.get(g[1:]), "norms": norms, "beta": beta,
                                         "length": ls, "params": p,
                                         "files": ["synthetic_tmi_80m.csv", "synthetic_dem.tif", *files]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    xs, ys, z, h = ground_grid(rng)
    gi = RegularGridInterpolator((ys, xs), z)
    G = lambda x, y: gi(np.column_stack([np.ravel(y), np.ravel(x)])).reshape(np.shape(x))   # noqa: E731
    with rasterio.open(OUT / "synthetic_dem.tif", "w", driver="GTiff", height=len(ys), width=len(xs), count=1,
                       dtype="float32", crs="EPSG:32643",
                       transform=from_origin(xs[0] - h / 2, ys[-1] + h / 2, h, h)) as dst:
        dst.write(z[::-1].astype("float32"), 1)
    sx = np.arange(X0, X0 + 5000 + 1, 50.0)
    SX, SY = np.meshgrid(sx, sx + (Y0 - X0))
    gs = G(SX, SY)
    cube, dyke = bodies(G)

    # the true model on 25 m cells around the bodies
    d = 25.0
    mx, my = np.arange(X0 + 500, X0 + 4600, d), np.arange(Y0 + 1000, Y0 + 4000, d)
    zmax, zmin = float(z.max()) + d, float(z.min()) - 1100.0
    nz = int(np.ceil((zmax - zmin) / d))
    mesh = TensorMesh([np.full(len(mx), d), np.full(len(my), d), np.full(nz, d)], origin=[mx[0], my[0], zmax - nz * d])
    model, in_cube, in_dyke = true_model(mesh.cell_centers, G, cube, dyke)
    act = model > 0

    # TMI 80 m above the ground, noise 2 % + 5 nT
    st = np.column_stack([SX.ravel(), SY.ravel(), gs.ravel() + 80.0])
    rx = mag.receivers.Point(st, components="tmi")
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], amplitude=FIELD[0], inclination=FIELD[1],
                                             declination=FIELD[2])
    sim = mag.Simulation3DIntegral(mesh, survey=mag.Survey(src), chiMap=maps.IdentityMap(nP=int(act.sum())),
                                   active_cells=act, store_sensitivities="forward_only")
    clean = sim.dpred(model[act])
    noisy = clean + rng.normal(0, 0.02 * np.abs(clean) + 5.0)
    np.savetxt(OUT / "synthetic_tmi_80m.csv", np.c_[st, noisy], delimiter=",", header="x,y,z,tmi", comments="", fmt="%.3f")
    np.savez_compressed(OUT / "truth_fine.npz", x=mx, y=my, origin=mesh.origin, nz=nz, h=d,
                        model=model.astype(np.float32))
    (OUT / "truth.json").write_text(json.dumps(
        {"crs": "EPSG:32643", "field": list(FIELD), "cube": cube, "dyke": dyke, "noise": "2 % + 5 nT",
         "station_height": 80.0, "station_spacing": 50.0, "seed": SEED,
         "dem": "synthetic_dem.tif (50 m, random seed 2026, about ±150 m)",
         "n_stations": int(len(noisy)), "tmi_range": [float(noisy.min()), float(noisy.max())],
         "ground_range": [float(gs.min()), float(gs.max())],
         "true_cells": {"cube": int(in_cube.sum()), "dyke": int(in_dyke.sum())},
         "true_volume_m3": {"cube": float(in_cube.sum() * d ** 3), "dyke": float(in_dyke.sum() * d ** 3)}},
        indent=1))

    (OUT / "spec_B.json").write_text(json.dumps(prior_B(cube, dyke, 10.0), indent=1))
    # the logs see the ground as the inversion does: the GeoTIFF's float32 heights
    g32 = RegularGridInterpolator((ys, xs), z.astype(np.float32).astype(float))
    G32 = lambda x, y: g32(np.column_stack([np.ravel(y), np.ravel(x)])).reshape(np.shape(x))   # noqa: E731
    rows, spec_c = boreholes_C(G32, cube, dyke)
    with open(OUT / "boreholes_C.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "spec_C.json").write_text(json.dumps(spec_c, indent=1))
    r = runs(cube, dyke, spec_c)
    (OUT / "runs.json").write_text(json.dumps(r, indent=1))
    hits = [f"{x['hole'].split()[0]} {x['unit']}" for x in rows if x["unit"] != "hole: background"]
    print(f"ground {gs.min():.0f}-{gs.max():.0f} m; TMI {noisy.min():.0f} to {noisy.max():.0f} nT at {len(noisy)} stations;"
          f" truth: cube {in_cube.sum()} and intrusion {in_dyke.sum()} cells of 25 m; holes hit: {hits}; {len(r)} runs")


if __name__ == "__main__":
    main()
