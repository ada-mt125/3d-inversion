"""Models of the Karnataka runs with terrain: grids, depth below the ground, metrics.

Used by the figure and report scripts of karnataka_gravity_terrain, karnataka_magnetic and
karnataka_joint.  Depths are measured below the ground of each column (the top of its
highest ground cell), not below a datum: the ground lies at 380-1090 m.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SHARED = Path(__file__).resolve().parent
INPUTS = SHARED.parent                    # examples/output/karnataka_inputs
REPO = INPUTS.parents[2]
sys.path.insert(0, str(REPO))

from geoinv3d.cloud.meshing import padding_cells  # noqa: E402
from geoinv3d.viz.result_workflow import load_result, result_mesh  # noqa: E402

AOI = (641000.0, 711000.0, 1634000.0, 1704000.0)
MAIN_HIGH = (671500.0, 1664500.0)         # the main Bouguer high
NW_HIGH = (650500.0, 1682500.0)


def prep():
    """Numbers and grids of prepare_inputs.py."""
    return json.loads((INPUTS / "prep.json").read_text(encoding="utf-8")), np.load(INPUTS / "prep.npz")


class Grid:
    """One property model of a run on its tensor mesh."""

    def __init__(self, meta, model=None):
        self.meta = meta
        mesh = self.mesh = result_mesh(meta)
        shape = self.shape = tuple(mesh.shape_cells)
        active = meta.get("_active")
        values = np.asarray(meta["_model"] if model is None else model, dtype=float)
        full = np.zeros(mesh.n_cells)
        if active is not None:
            full[active] = values
            self.active = np.asarray(active, bool).reshape(shape, order="F")
        else:
            full[:] = values
            self.active = np.ones(shape, bool)
        self.m = full.reshape(shape, order="F")
        self.vol = mesh.cell_volumes.reshape(shape, order="F")
        used = meta["mesh_design"]["used"]
        self.n_pad = padding_cells(used["core_cell_m"], used["pad_distance_m"])
        nx, ny, nz = shape
        self.sx, self.sy = slice(self.n_pad, nx - self.n_pad), slice(self.n_pad, ny - self.n_pad)
        self.zc = mesh.cell_centers_z
        self.dz = mesh.h[2]
        self.z_core_base = float(mesh.nodes_z[self.n_pad])       # below it: bottom padding
        # the ground of each column: the top of its highest ground cell
        top = nz - 1 - np.argmax(self.active[:, :, ::-1], axis=2)
        self.ground = mesh.nodes_z[top + 1]
        self.depth = self.ground[:, :, None] - self.zc[None, None, :]     # of cell centres
        core = np.zeros((nx, ny), bool)
        core[self.sx, self.sy] = True
        self.core_xy = core
        self.ground_mean = float(self.ground[core].mean())
        self.core_depth = float(used["depth_core_m"])

    def ij(self, xy):
        return (int(np.argmin(np.abs(self.mesh.cell_centers_x - xy[0]))),
                int(np.argmin(np.abs(self.mesh.cell_centers_y - xy[1]))))

    def shares(self):
        """Where the anomalous mass (|m| x volume) lies: core, lateral padding, below the core;
        and the depths below the ground above which half and 90 % of the core columns' lie."""
        a = np.abs(self.m) * self.vol * self.active
        below = np.broadcast_to((self.zc < self.z_core_base)[None, None, :], a.shape)
        core = self.core_xy[:, :, None] & ~below
        tot = a.sum()
        out = {"core": float(a[core].sum() / tot), "below": float(a[below].sum() / tot)}
        out["lateral"] = 1.0 - out["core"] - out["below"]
        cols = self.core_xy[:, :, None] & self.active
        d, w = self.depth[cols], a[cols]
        o = np.argsort(d)
        cum = np.cumsum(w[o]) / w.sum()
        out["D50"] = float(np.interp(0.5, cum, d[o]) / 1e3)
        out["D90"] = float(np.interp(0.9, cum, d[o]) / 1e3)
        return out

    def column(self, xy):
        """Profile under (x, y): depth below the ground (km, ascending) and values."""
        i, j = self.ij(xy)
        k = np.where(self.active[i, j])[0][::-1]
        return self.depth[i, j, k] / 1e3, self.m[i, j, k], self.dz[k] / 1e3

    def integrated(self, absolute=False):
        """Property x thickness summed down each core column (unit x km), and the centroid
        depth of |property| (km below the ground; NaN where the column holds little)."""
        m = self.m[self.sx, self.sy] * self.active[self.sx, self.sy]
        dz = self.dz[None, None, :]
        total = ((np.abs(m) if absolute else m) * dz).sum(axis=2) / 1e3
        a = (np.abs(m) * dz).sum(axis=2)
        cen = (np.abs(m) * dz * self.depth[self.sx, self.sy]).sum(axis=2) / np.maximum(a, 1e-30) / 1e3
        cen[a < 0.2 * a.max()] = np.nan
        return total, cen

    def mass_profile(self, edges_km):
        """Share of the core columns' |mass| per km of depth below the ground."""
        a = (np.abs(self.m) * self.vol * self.active)[self.sx, self.sy]
        d = self.depth[self.sx, self.sy] / 1e3
        half = (self.dz / 2e3)[None, None, :]
        top, thick = d - half, 2 * half
        # each cell's mass spread over its own depth range (the padding layers are thick)
        cum = np.array([(a * np.clip((e - top) / thick, 0.0, 1.0)).sum() for e in edges_km])
        return np.diff(cum) / a.sum() / np.diff(edges_km)

    def section(self, northing):
        """E-W section of the core at ``northing``: x nodes (km), z nodes (km), values
        (nz, nx) with air as NaN, and the ground along it (km)."""
        j = int(np.argmin(np.abs(self.mesh.cell_centers_y - northing)))
        v = np.where(self.active[self.sx, j, :], self.m[self.sx, j, :], np.nan).T
        return (self.mesh.nodes_x[self.n_pad:self.shape[0] - self.n_pad + 1] / 1e3,
                self.mesh.nodes_z / 1e3, v, self.ground[self.sx, j] / 1e3,
                float(self.mesh.cell_centers_y[j]))

    def depth_slice(self, depth_km):
        """The core at about ``depth_km`` below the mean ground: x, y nodes (km), values."""
        k = int(np.argmin(np.abs(self.ground_mean - self.zc - depth_km * 1e3)))
        v = np.where(self.active[self.sx, self.sy, k], self.m[self.sx, self.sy, k], np.nan)
        nx, ny, _ = self.shape
        return (self.mesh.nodes_x[self.n_pad:nx - self.n_pad + 1] / 1e3,
                self.mesh.nodes_y[self.n_pad:ny - self.n_pad + 1] / 1e3, v,
                float((self.ground_mean - self.zc[k]) / 1e3))


def body(z, v, dz):
    """Peak, half-maximum depth range and centroid depth of a positive body in a profile.

    Models held at the upper bound have a flat peak, so the depth of "the" peak is not
    defined; the centroid (positive values x thickness) is.
    """
    if v.max() <= 0:
        return {"peak": float(v.max()), "top_km": float("nan"), "bottom_km": float("nan"),
                "centroid_km": float("nan")}
    inside = v >= v.max() / 2
    w = np.clip(v, 0, None) * dz
    return {"peak": float(v.max()), "top_km": float((z - dz / 2)[inside].min()),
            "bottom_km": float((z + dz / 2)[inside].max()), "centroid_km": float((w @ z) / w.sum())}


def data_fit(d):
    r = d["observed"] - d["predicted"]
    return {"chi2": float(np.mean((r / d["std"]) ** 2)), "rms": float(np.sqrt(np.mean(r ** 2))),
            "max_abs": float(np.abs(r).max()),
            "corr": float(np.corrcoef(d["observed"], d["predicted"])[0, 1])}


def gridded(locs, values):
    """Station values on their regular grid (km axes)."""
    xs, ys = np.unique(np.round(locs[:, 0], 1)), np.unique(np.round(locs[:, 1], 1))
    g = np.full((len(ys), len(xs)), np.nan)
    g[np.searchsorted(ys, np.round(locs[:, 1], 1)), np.searchsorted(xs, np.round(locs[:, 0], 1))] = values
    return xs / 1e3, ys / 1e3, g


def load(path):
    return load_result(path)


def box_stats(g: Grid, box):
    """|property| x volume inside ``box`` = (west, east, south, north), over all depths:
    total (unit x km3), centroid depth and the depths above which 10, 50 and 90 % lie (km
    below the ground), and the largest value."""
    cx, cy = g.mesh.cell_centers_x, g.mesh.cell_centers_y
    sel = ((cx >= box[0]) & (cx <= box[1]))[:, None] & ((cy >= box[2]) & (cy <= box[3]))[None, :]
    cols = sel[:, :, None] & g.active
    a = (np.abs(g.m) * g.vol)[cols]
    d = g.depth[cols]
    o = np.argsort(d)
    cum = np.cumsum(a[o]) / a.sum()
    q = [float(np.interp(p, cum, d[o]) / 1e3) for p in (0.1, 0.5, 0.9)]
    peak = float(np.abs(g.m[cols]).max())
    return {"total": float(a.sum() / 1e9), "centroid_km": float((a @ d) / a.sum() / 1e3),
            "d10_km": q[0], "d50_km": q[1], "d90_km": q[2], "max": peak,
            # the volume of the cells above half the largest value
            "volume_half_km3": float(g.vol[cols][np.abs(g.m[cols]) >= 0.5 * peak].sum() / 1e9)}
