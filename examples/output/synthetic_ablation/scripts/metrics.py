"""The true model of the ablation and the measures of a result against it.

Used by ``make_figures.py``.  Every measure is over the core: the 5 × 5 km survey, down to
2 km below the ground, weighted by cell volume.

    corr        correlation of the model with the true model
    k_cube      mean susceptibility in the true cube (true 0.5 SI)
    k_dyke      mean susceptibility in the true intrusion (true 1 SI)
    cube_depth  depth of the centre of magnetization (κ × volume) in the cube's column, 600 × 600 m
                (true 250 m: the cube is 100–400 m below the ground)
    dip         the intrusion's apparent dip: the slope of the magnetization's mean easting against
                depth, 100–900 m, in its middle 1.4 km along strike (true 45°)
    ghost       share of the magnetization outside the two true bodies
    k_wrong     mean susceptibility in the prior bodies of group B where there is no true body
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
T = json.loads((INPUTS / "truth.json").read_text())
CUBE, DYKE = T["cube"], T["dyke"]
SPEC_B = json.loads((INPUTS / "spec_B.json").read_text())
HOLES = list(csv.DictReader((INPUTS / "boreholes_C.csv").read_text(encoding="utf-8").splitlines()))
CORE = (600000.0, 605000.0, 1600000.0, 1605000.0)
BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, np.inf)]

with rasterio.open(INPUTS / "synthetic_dem.tif") as _r:
    _z = _r.read(1)[::-1].astype(float)
    _t = _r.transform
    _xs = _t.c + _t.a * (np.arange(_r.width) + 0.5)
    _ys = _t.f + _t.e * _r.height - _t.e * (np.arange(_r.height) + 0.5)
_gi = RegularGridInterpolator((_ys, _xs), _z)


def ground(x, y):
    """The ground's elevation (m) at points x, y (any shape)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return _gi(np.column_stack([y.ravel(), x.ravel()])).reshape(x.shape)


def truth_at(cc):
    """(κ, in the cube, in the intrusion, depth below the ground) at points (x, y, elevation)."""
    half = CUBE["edge"] / 2
    inc = (np.abs(cc[:, 0] - CUBE["centre"][0]) <= half) & (np.abs(cc[:, 1] - CUBE["centre"][1]) <= half) \
        & (cc[:, 2] <= CUBE["top_elev"]) & (cc[:, 2] >= CUBE["bottom_elev"])
    depth = ground(cc[:, 0], cc[:, 1]) - cc[:, 2]
    hw = DYKE["thickness"] / np.sin(np.radians(DYKE["dip"])) / 2
    xc = DYKE["top_x"] + (depth - DYKE["top_depth"]) / np.tan(np.radians(DYKE["dip"]))
    ind = (cc[:, 1] >= DYKE["y"][0]) & (cc[:, 1] <= DYKE["y"][1]) & (depth >= DYKE["top_depth"]) \
        & (depth <= DYKE["bottom_depth"]) & (np.abs(cc[:, 0] - xc) <= hw)
    return np.where(ind, DYKE["kappa"], np.where(inc, CUBE["kappa"], 0.0)), inc, ind, depth


def prior_B_at(cc, depth):
    """Points inside the prior bodies of group B."""
    s0, s1 = SPEC_B["sources"]
    p = np.array(s0["polygon"])
    in_c = (cc[:, 0] >= p[:, 0].min()) & (cc[:, 0] <= p[:, 0].max()) & (cc[:, 1] >= p[:, 1].min()) \
        & (cc[:, 1] <= p[:, 1].max()) & (depth >= s0["top_m"]) & (depth <= s0["bottom_m"])
    q = np.array(s1["polygon"])
    shift = (depth - s1["top_m"]) / np.tan(np.radians(s1["dip"]))
    in_d = (cc[:, 0] - shift >= q[:, 0].min()) & (cc[:, 0] - shift <= q[:, 0].max()) \
        & (cc[:, 1] >= q[:, 1].min()) & (cc[:, 1] <= q[:, 1].max()) & (depth >= s1["top_m"]) & (depth <= s1["bottom_m"])
    return in_c | in_d


def hole_traces(step=5.0):
    """The holes, every ``step`` m: {name: (along-hole distance, x, y, z, true κ)}."""
    out, start = {}, {}
    for h in HOLES:            # one row per interval, in order down the hole
        L, inc, az = float(h["length_m"]), np.radians(float(h["inclination"])), np.radians(float(h["bearing"]))
        s = np.linspace(0.0, L, max(int(round(L / step)), 1) + 1)
        pts = np.column_stack([float(h["x"]) + s * np.cos(inc) * np.sin(az),
                               float(h["y"]) + s * np.cos(inc) * np.cos(az), float(h["z_top"]) - s * np.sin(inc)])
        prev = out.get(h["hole"])
        s0 = start.get(h["hole"], 0.0)
        start[h["hole"]] = s0 + L
        k = np.full(len(s), {"hole: background": 0.0, "hole: 0.5 SI": 0.5, "hole: 1 SI": 1.0}[h["unit"]])
        arr = (s + s0, pts[:, 0], pts[:, 1], pts[:, 2], k)
        out[h["hole"]] = arr if prev is None else tuple(np.concatenate([a, b]) for a, b in zip(prev, arr))
    return out


class Cells:
    """The active cells of a result: centres, volumes, the truth there and the core."""

    def __init__(self, mesh, active):
        self.cc, self.vol = mesh.cell_centers[active], mesh.cell_volumes[active]
        self.t, self.inc, self.ind, self.depth = truth_at(self.cc)
        x0, x1, y0, y1 = CORE
        self.core = (self.cc[:, 0] >= x0) & (self.cc[:, 0] <= x1) & (self.cc[:, 1] >= y0) & (self.cc[:, 1] <= y1) \
            & (self.depth <= 2000)
        self.wrong = prior_B_at(self.cc, self.depth) & ~self.inc & ~self.ind
        from scipy.spatial import cKDTree
        pts = np.vstack([np.column_stack(v[1:4]) for v in hole_traces().values()])
        self.hole_dist = cKDTree(pts).query(self.cc)[0]


def measures(m, c: Cells) -> dict:
    """The measures of a model ``m`` on the cells ``c``."""
    vol, core = c.vol, c.core
    w = vol[core]
    mc, tc = m[core], c.t[core]
    mu = lambda a: (a * w).sum() / w.sum()   # noqa: E731
    corr = mu((mc - mu(mc)) * (tc - mu(tc))) / np.sqrt(mu((mc - mu(mc)) ** 2) * mu((tc - mu(tc)) ** 2))
    mean = lambda s: float((m[s] * vol[s]).sum() / vol[s].sum()) if s.any() else float("nan")   # noqa: E731
    mass = m * vol
    cx, cy = CUBE["centre"]
    col = core & (np.abs(c.cc[:, 0] - cx) <= 300) & (np.abs(c.cc[:, 1] - cy) <= 300)
    zone = core & (c.cc[:, 1] >= DYKE["y"][0] + 300) & (c.cc[:, 1] <= DYKE["y"][1] - 300) \
        & (c.cc[:, 0] >= DYKE["top_x"] - 600) & (c.cc[:, 0] <= DYKE["top_x"] + 1600)
    zs, xm = [], []
    for d0 in np.arange(100, 900, 100):
        s = zone & (c.depth >= d0) & (c.depth < d0 + 100)
        if mass[s].sum() > 0:
            zs.append(d0 + 50)
            xm.append((mass[s] * c.cc[s, 0]).sum() / mass[s].sum())
    slope = np.polyfit(zs, xm, 1)[0] if len(zs) >= 3 else np.nan       # metres east per metre down
    return {"corr": float(corr), "k_cube": mean(c.inc), "k_dyke": mean(c.ind),
            "cube_depth": float((mass[col] * c.depth[col]).sum() / mass[col].sum()),
            "dip": float(np.degrees(np.arctan2(1.0, slope))) if np.isfinite(slope) else None,
            "ghost": float(mass[core & ~c.inc & ~c.ind].sum() / mass[core].sum()),
            "k_wrong": mean(c.wrong)}


def by_distance(values, c: Cells) -> list:
    """Volume-weighted mean of ``values`` over the core in the distance bands from the holes."""
    out = []
    for lo, hi in BANDS:
        s = c.core & (c.hole_dist >= lo) & (c.hole_dist < hi)
        out.append(float((values[s] * c.vol[s]).sum() / c.vol[s].sum()))
    return out
