"""Slices of a result's model at the resolution of its own mesh.

The 3D tab's grid is the mesh's core resampled and, for large meshes, coarsened (at most
MAX_VIEWER_CELLS voxels: 40 m cells over 5 km come out at 80 m).  The depth slice and the
sections are instead sampled here from the mesh itself, at half its finest cells, on a
plane at a depth below the ground (or an elevation), along an E-W or N-S line or along any
profile.  Cells above the ground (inactive) are None.

A ``ResultModel`` wraps one result (``load_result``): its mesh, its model on every cell (an
MVI result's is the magnetization amplitude), and its ground (topography.npz, or the flat
elevation).
"""

from __future__ import annotations

import math

import numpy as np

MAX_SIDE = 400        # samples along a section or a side of a depth slice
MAX_DEPTH_STEPS = 300


def _round(a, digits=4):
    """Values to ``digits`` significant figures, NaN as None (for JSON)."""
    out = []
    for row in np.asarray(a, dtype=float):
        out.append([None if not math.isfinite(v) else float(f"{v:.{digits}g}") for v in row])
    return out


class ResultModel:
    def __init__(self, meta):
        from .result_workflow import result_mesh
        self.meta = meta
        self.mesh = mesh = result_mesh(meta)
        active = meta.get("_active")
        model = np.asarray(meta["_model"], dtype=float)
        full = np.full(mesh.n_cells, np.nan)
        if active is None:
            full[:] = model
        else:
            full[np.asarray(active, bool)] = model
        self.values = full
        used = meta["mesh_design"]["used"]
        self.h, self.dz = float(used["core_cell_m"]), float(used["core_cell_z_m"])
        self.depth_core = float(used["depth_core_m"])
        ex = meta.get("cell_centers_x", [0, 0]) + meta.get("cell_centers_y", [0, 0])
        self.extent = [float(v) for v in ex]          # the stations' extent
        finite = full[np.isfinite(full)]
        lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
        self.range = [float(lo), float(hi)]
        self.signed = bool(finite.size and finite.min() < -1e-9 * max(1.0, abs(finite.max())))
        topo = meta.get("_topography")
        if topo is not None and np.size(topo.get("z", [])) > 1:
            from scipy.interpolate import RegularGridInterpolator
            gx, gy, gz = np.asarray(topo["x"]), np.asarray(topo["y"]), np.asarray(topo["z"], float)
            interp = RegularGridInterpolator((gy, gx), gz, bounds_error=False, fill_value=None)
            self.ground = lambda x, y: interp(np.column_stack([np.ravel(y), np.ravel(x)])).reshape(np.shape(x))
        else:
            flat = float((meta.get("topography") or {}).get("flat_elevation", 0.0) or 0.0)
            self.ground = lambda x, y: np.full(np.shape(x), flat)
        nodes = mesh.nodes
        self._lo, self._hi = nodes.min(axis=0), nodes.max(axis=0)

    # ── sampling ──
    def sample(self, pts) -> np.ndarray:
        """The model at (n, 3) points; NaN outside the mesh or above the ground."""
        pts = np.asarray(pts, dtype=float)
        out = np.full(len(pts), np.nan)
        inside = np.all((pts >= self._lo) & (pts <= self._hi), axis=1)
        if inside.any():
            out[inside] = self.values[self._cells(pts[inside])]
        return out

    def _cells(self, pts) -> np.ndarray:
        from discretize import TensorMesh
        m = self.mesh
        if isinstance(m, TensorMesh):
            idx = [np.clip(np.searchsorted(n, p, side="right") - 1, 0, len(n) - 2)
                   for n, p in zip((m.nodes_x, m.nodes_y, m.nodes_z), pts.T)]
            return np.ravel_multi_index(idx, m.shape_cells, order="F")
        return m.get_containing_cells(pts)

    def _depth_axis(self, depth_max, top_offset=0.0):
        step = max(self.dz / 2.0, depth_max / MAX_DEPTH_STEPS)
        return np.arange(top_offset + step / 2.0, depth_max, step)

    # ── a vertical section along a line ──
    def line(self, x0, y0, x1, y1, ref: str = "ground", depth_max=None) -> dict:
        """Along A (x0, y0) to B (x1, y1): ``ref`` "ground" gives depth below the ground on
        the vertical axis (the ground at 0), "elev" elevation (cells above the ground None)."""
        length = math.hypot(x1 - x0, y1 - y0)
        if not length > 0:
            raise ValueError("The profile needs two different points")
        depth_max = float(depth_max or self.depth_core)
        step = max(self.h / 2.0, length / MAX_SIDE)
        s = np.arange(step / 2.0, length, step)
        x, y = x0 + (x1 - x0) * s / length, y0 + (y1 - y0) * s / length
        ground = self.ground(x, y)
        if ref == "ground":
            v = self._depth_axis(depth_max)
            z = ground[None, :] - v[:, None]
        else:
            top, bottom = float(np.max(ground)), float(np.min(ground)) - depth_max
            vstep = max(self.dz / 2.0, (top - bottom) / MAX_DEPTH_STEPS)
            v = np.arange(top - vstep / 2.0, bottom, -vstep)
            z = np.broadcast_to(v[:, None], (len(v), len(s)))
        X, Y = np.broadcast_to(x, z.shape), np.broadcast_to(y, z.shape)
        vals = self.sample(np.column_stack([X.ravel(), Y.ravel(), z.ravel()])).reshape(z.shape)
        if ref != "ground":
            vals[z > ground[None, :]] = np.nan      # the air
        return {"kind": "line", "ref": ref, "a": [x0, y0], "b": [x1, y1], "length": length,
                "h": [float(t) for t in s], "v": [float(t) for t in v], "ground": [float(t) for t in ground],
                "values": _round(vals), **self.info()}

    # ── a depth slice ──
    def plan(self, level: float, ref: str = "ground", extent=None) -> dict:
        """A horizontal slice over the stations' extent: ``level`` metres below the ground
        (``ref`` "ground") or an elevation (``ref`` "elev")."""
        W, E, S, N = extent or self.extent
        step = max(self.h / 2.0, max(E - W, N - S) / MAX_SIDE)
        xs, ys = np.arange(W + step / 2.0, E, step), np.arange(S + step / 2.0, N, step)
        X, Y = np.meshgrid(xs, ys)
        z = self.ground(X, Y) - level if ref == "ground" else np.full(X.shape, float(level))
        vals = self.sample(np.column_stack([X.ravel(), Y.ravel(), z.ravel()])).reshape(X.shape)
        return {"kind": "plan", "ref": ref, "level": float(level), "x": [float(t) for t in xs],
                "y": [float(t) for t in ys], "values": _round(vals), **self.info()}

    def info(self) -> dict:
        return {"range": self.range, "signed": self.signed, "cell_m": self.h, "cell_z_m": self.dz}

    # ── the strike, for a default profile across it ──
    def strike_profile(self, level: float = None) -> dict:
        """A profile through the strongest bodies, across their strike: the principal axis of
        the cells above half the 98th percentile on a depth slice (a third of the core deep
        unless ``level``), the profile perpendicular to it through their centre, as long as
        the stations' extent allows."""
        level = self.depth_core / 6.0 if level is None else float(level)
        p = self.plan(level)
        vals = np.array([[np.nan if v is None else v for v in row] for row in p["values"]], float)
        a = np.abs(vals)
        X, Y = np.meshgrid(p["x"], p["y"])
        W, E, S, N = self.extent
        cx, cy = (W + E) / 2.0, (S + N) / 2.0
        ok = np.isfinite(a)
        if ok.sum() > 10 and np.nanmax(a) > 0:
            thr = 0.5 * np.nanpercentile(a[ok], 98)
            sel = ok & (a >= thr)
            if sel.sum() >= 5:
                w = a[sel]
                cx, cy = float(np.average(X[sel], weights=w)), float(np.average(Y[sel], weights=w))
                dx, dy = X[sel] - cx, Y[sel] - cy
                cov = np.cov(np.vstack([dx, dy]), aweights=w)
                evals, evecs = np.linalg.eigh(cov)
                sx, sy = evecs[:, -1]            # the strike direction
                ux, uy = -sy, sx                 # across it
            else:
                ux, uy = 1.0, 0.0
        else:
            ux, uy = 1.0, 0.0
        if ux < 0 or (ux == 0 and uy < 0):       # A to the west (or south)
            ux, uy = -ux, -uy
        # as far as the extent goes on both sides of the centre
        reach = []
        for t in (1, -1):
            lim = []
            for c, u, lo, hi in ((cx, t * ux, W, E), (cy, t * uy, S, N)):
                if abs(u) > 1e-12:
                    lim.append(((hi if u > 0 else lo) - c) / u)
            reach.append(max(0.0, min(lim)) if lim else 0.0)
        r_pos, r_neg = reach
        return {"a": [cx - r_neg * ux, cy - r_neg * uy], "b": [cx + r_pos * ux, cy + r_pos * uy],
                "centre": [cx, cy], "strike_deg": float(math.degrees(math.atan2(uy, -ux)) % 180.0)}


# ── Several runs of one area: where their models agree ──
BODY_SHARE = 0.25          # a body: above 25 % of the run's 98th percentile (of its positive values)
ROBUST_SHARE = 0.8         # robust: a body in at least 80 % of the runs


def body_level(model: ResultModel) -> float:
    """A body's threshold for a run: BODY_SHARE of the 98th percentile of its positive values
    in the core (under the stations, down to the core depth), so that what the padding holds
    (often large values in large cells) does not set it."""
    # sampled evenly through the core, not cell by cell: an octree's cells are mostly the
    # small ones near the surface, which would weigh a compact model's threshold 2-3 times up
    # and a smooth one's (deep, in large cells) down
    if getattr(model, "_body_level", None) is None:
        W, E, S, N = model.extent
        X, Y = np.meshgrid(np.linspace(W, E, 48), np.linspace(S, N, 48))
        G = model.ground(X, Y).ravel()
        ds = np.linspace(25.0, model.depth_core, 40)
        pts = np.column_stack([np.tile(X.ravel(), len(ds)), np.tile(Y.ravel(), len(ds)),
                               (G[None, :] - ds[:, None]).ravel()])
        v = model.sample(pts)
        v = v[np.isfinite(v) & (v > 0)]
        model._body_level = BODY_SHARE * float(np.percentile(v, 98)) if v.size else np.inf
    return model._body_level


def agreement(models, pts) -> np.ndarray:
    """The share of ``models`` with a body at each of the (n, 3) points (NaN where none of
    them has a model there, e.g. above the ground)."""
    pts = np.asarray(pts, dtype=float)
    hits = np.zeros(len(pts))
    seen = np.zeros(len(pts))
    for m in models:
        v = m.sample(pts)
        ok = np.isfinite(v)
        seen += ok
        hits += ok & (v >= body_level(m))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, hits / np.maximum(seen, 1), np.nan)


def agreement_slice(base: ResultModel, models, kind: str, **kw) -> dict:
    """A slice of the agreement, on the geometry ``base.line`` / ``base.plan`` would give."""
    d = base.line(**kw) if kind == "line" else base.plan(**kw)
    if kind == "line":
        L = d["length"]
        (x0, y0), (x1, y1) = d["a"], d["b"]
        h, v = np.asarray(d["h"]), np.asarray(d["v"])
        x, y = x0 + (x1 - x0) * h / L, y0 + (y1 - y0) * h / L
        ground = np.asarray(d["ground"])
        z = ground[None, :] - v[:, None] if d["ref"] == "ground" else np.broadcast_to(v[:, None], (len(v), len(h)))
        X, Y = np.broadcast_to(x, z.shape), np.broadcast_to(y, z.shape)
        a = agreement(models, np.column_stack([X.ravel(), Y.ravel(), np.ravel(z)])).reshape(z.shape)
        if d["ref"] != "ground":
            a[z > ground[None, :]] = np.nan
    else:
        X, Y = np.meshgrid(d["x"], d["y"])
        z = base.ground(X, Y) - d["level"] if d["ref"] == "ground" else np.full(X.shape, d["level"])
        a = agreement(models, np.column_stack([X.ravel(), Y.ravel(), z.ravel()])).reshape(X.shape)
    return {**d, "values": _round(a, 3), "range": [0.0, 1.0], "signed": False,
            "field": "agreement", "n_runs": len(models)}


def agreement_grid(base_meta, models) -> dict:
    """The agreement on the base run's 3D grid (result_workflow.ViewerGrid), in its order."""
    from .result_workflow import ViewerGrid, result_mesh
    g = ViewerGrid(base_meta, result_mesh(base_meta)).geometry()
    xe, ye, ze = (np.asarray(g[k]) for k in ("x_edges", "y_edges", "z_edges"))   # z top down
    xc, yc, zc = (0.5 * (e[1:] + e[:-1]) for e in (xe, ye, ze))
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")          # x fastest, then y, then z
    a = agreement(models, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    return {**g, "values": [None if not math.isfinite(v) else round(float(v), 3) for v in a]}


def ensemble_summary(base: ResultModel, models, depths=(100, 300, 600, 1000)) -> dict:
    """Where the runs agree at some depths below the ground, and how deep each puts its
    bodies: the moment-weighted mean depth of its model in the columns where at least 80 %
    of the runs have a body in the top kilometre."""
    W, E, S, N = base.extent
    step = max(base.h, max(E - W, N - S) / 120.0)
    xs, ys = np.arange(W + step / 2, E, step), np.arange(S + step / 2, N, step)
    X, Y = np.meshgrid(xs, ys)
    G = base.ground(X, Y)
    zs = np.arange(25.0, min(base.depth_core, 3000.0), 50.0)
    pts = np.column_stack([np.repeat(X.ravel()[None, :], len(zs), 0).ravel(),
                           np.repeat(Y.ravel()[None, :], len(zs), 0).ravel(),
                           (G.ravel()[None, :] - zs[:, None]).ravel()])
    vols = [m.sample(pts).reshape(len(zs), -1) for m in models]
    bodies = [np.isfinite(v) & (v >= body_level(m)) for v, m in zip(vols, models)]
    frac = np.mean(bodies, axis=0)                       # (depths, columns)
    out = {"n_runs": len(models), "depths": []}
    for d in depths:
        k = int(np.argmin(abs(zs - d)))
        f = frac[k]
        out["depths"].append({"depth_m": float(d), "robust_share": float(np.mean(f >= ROBUST_SHARE)),
                              "any_share": float(np.mean(f > 0))})
    top = np.mean([b[zs <= 1000].any(axis=0) for b in bodies], axis=0) >= ROBUST_SHARE
    out["robust_columns_share"] = float(np.mean(top))
    centres = []
    for v in vols:
        w = np.clip(np.nan_to_num(v[:, top]), 0, None)
        centres.append(float(np.median((w * zs[:, None]).sum(0) / np.maximum(w.sum(0), 1e-12)))
                       if top.any() else None)
    out["moment_depth_m"] = centres
    return out
