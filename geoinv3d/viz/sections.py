"""Slices of a result's model at the resolution of its own mesh.

The 3D tab's grid is the mesh's core resampled and, for large meshes, coarsened (at most
MAX_VIEWER_CELLS voxels: 40 m cells over 5 km come out at 80 m).  The depth slice and the
sections are instead sampled here from the mesh itself, at half its finest cells, on a
plane at a depth below the ground (or an elevation), along an E-W or N-S line or along any
profile.  Cells above the ground (inactive) are None.

A ``ResultModel`` wraps one result (``load_result``): its mesh, its model on every cell (an
MVI result's is the magnetization amplitude), its ground (topography.npz, or the flat
elevation), and for a Bayesian posterior its standard deviation, the probability of a
body and the samples (``fields``; a profile of the probability carries the depths of the
bodies' top and base in the samples).
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
        from .result_workflow import result_mesh, robust_range
        self.meta = meta
        self.mesh = mesh = result_mesh(meta)
        active = meta.get("_active")
        model = np.asarray(meta["_model"], dtype=float)
        self._active = None if active is None else np.asarray(active, bool)
        full = self._on_mesh(model)
        self.values = full
        # a Bayesian posterior (worker.run_bayesian_inversion): its standard deviation, the
        # probability of a body, and the samples (for the depths of the bodies)
        self.fields = {"model": full}
        if meta.get("_posterior_std") is not None:
            self.fields["std"] = self._on_mesh(meta["_posterior_std"])
        if meta.get("_prob_body") is not None:
            self.fields["probability"] = self._on_mesh(meta["_prob_body"])
        self.samples = meta.get("_samples")
        self.threshold = (meta.get("bayes") or {}).get("threshold")
        self._probs = {}
        used = meta["mesh_design"]["used"]
        self.h, self.dz = float(used["core_cell_m"]), float(used["core_cell_z_m"])
        self.depth_core = float(used["depth_core_m"])
        ex = meta.get("cell_centers_x", [0, 0]) + meta.get("cell_centers_y", [0, 0])
        self.extent = [float(v) for v in ex]          # the stations' extent
        finite = full[np.isfinite(full)]
        self.range = robust_range(finite)          # the 3D view's too
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

    def _on_mesh(self, v):
        full = np.full(self.mesh.n_cells, np.nan)
        if self._active is None:
            full[:] = np.asarray(v, dtype=float)
        else:
            full[self._active] = np.asarray(v, dtype=float)
        return full

    # ── sampling ──
    def cells_at(self, pts):
        """(inside, cells): which of the (n, 3) points are in the mesh, and their cells."""
        pts = np.asarray(pts, dtype=float)
        inside = np.all((pts >= self._lo) & (pts <= self._hi), axis=1)
        return inside, (self._cells(pts[inside]) if inside.any() else np.zeros(0, int))

    def probability(self, threshold=None) -> np.ndarray:
        """The share of the posterior samples above ``threshold`` (default the run's), on
        every cell."""
        if threshold is None or self.samples is None:
            return self.fields["probability"]
        key = round(float(threshold), 9)
        if key not in self._probs:
            if len(self._probs) > 8:
                self._probs.clear()
            self._probs[key] = self._on_mesh((np.asarray(self.samples) >= key).mean(axis=0))
        return self._probs[key]

    def sample(self, pts, field: str = "model", threshold=None) -> np.ndarray:
        """A field at (n, 3) points; NaN outside the mesh or above the ground."""
        values = self.probability(threshold) if field == "probability" else self.fields[field]
        out = np.full(len(pts), np.nan)
        inside, cells = self.cells_at(pts)
        if inside.any():
            out[inside] = values[cells]
        return out

    def _field_info(self, field):
        if field == "model":
            return self.info()
        if field == "probability":
            return {**self.info(), "range": [0.0, 1.0], "signed": False, "field": field}
        v = self.fields[field]
        v = v[np.isfinite(v)]
        return {**self.info(), "range": [0.0, float(np.percentile(v, 99)) if v.size else 1.0],
                "signed": False, "field": field}

    def body_depths(self, pts_shape, pts, threshold=None) -> dict:
        """From the posterior samples, along a section's columns: the share of samples with a
        body there, and the 10/50/90 % of the first and last rows (top and base) where they
        have one (None where under a tenth of them do)."""
        thr = self.threshold if threshold is None else float(threshold)
        if self.samples is None or thr is None:
            return {}
        nv, nh = pts_shape
        inside, cells = self.cells_at(pts)
        idx = np.full(len(pts), -1)
        if self._active is not None:
            pos = np.full(self.mesh.n_cells, -1)
            pos[self._active] = np.arange(int(self._active.sum()))
            idx[inside] = pos[cells]
        else:
            idx[inside] = cells
        ok = idx >= 0
        tops, bases = [], []
        for smp in np.asarray(self.samples):
            v = np.full(len(pts), np.nan)
            v[ok] = smp[idx[ok]]
            body = (v >= thr).reshape(nv, nh)
            anyb = body.any(axis=0)
            first = np.where(anyb, body.argmax(axis=0), -1)
            last = np.where(anyb, nv - 1 - body[::-1].argmax(axis=0), -1)
            tops.append(first)
            bases.append(last)
        tops, bases = np.array(tops), np.array(bases)
        share = (tops >= 0).mean(axis=0)
        def q(a, f):
            out = []
            for j in range(nh):
                col = a[:, j][a[:, j] >= 0]
                out.append(int(np.percentile(col, f)) if share[j] >= 0.1 and col.size else None)
            return out
        return {"body_share": [round(float(x), 3) for x in share],
                "top_rows": {str(f): q(tops, f) for f in (10, 50, 90)},
                "base_rows": {str(f): q(bases, f) for f in (10, 50, 90)}}

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
    def line(self, x0, y0, x1, y1, ref: str = "ground", depth_max=None, field: str = "model",
             threshold=None) -> dict:
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
        pts = np.column_stack([X.ravel(), Y.ravel(), np.ravel(z)])
        vals = self.sample(pts, field, threshold).reshape(z.shape)
        if ref != "ground":
            vals[z > ground[None, :]] = np.nan      # the air
        out = {"kind": "line", "ref": ref, "a": [x0, y0], "b": [x1, y1], "length": length,
               "h": [float(t) for t in s], "v": [float(t) for t in v], "ground": [float(t) for t in ground],
               "values": _round(vals, 3 if field == "probability" else 4), **self._field_info(field)}
        if field == "probability":
            out["depths"] = self.body_depths(z.shape, pts, threshold)
            out["threshold"] = self.threshold if threshold is None else float(threshold)
        return out

    # ── a depth slice ──
    def plan(self, level: float, ref: str = "ground", extent=None, field: str = "model",
             threshold=None) -> dict:
        """A horizontal slice over the stations' extent: ``level`` metres below the ground
        (``ref`` "ground") or an elevation (``ref`` "elev")."""
        W, E, S, N = extent or self.extent
        step = max(self.h / 2.0, max(E - W, N - S) / MAX_SIDE)
        xs, ys = np.arange(W + step / 2.0, E, step), np.arange(S + step / 2.0, N, step)
        X, Y = np.meshgrid(xs, ys)
        z = self.ground(X, Y) - level if ref == "ground" else np.full(X.shape, float(level))
        vals = self.sample(np.column_stack([X.ravel(), Y.ravel(), z.ravel()]), field, threshold).reshape(X.shape)
        return {"kind": "plan", "ref": ref, "level": float(level), "x": [float(t) for t in xs],
                "y": [float(t) for t in ys], "values": _round(vals, 3 if field == "probability" else 4),
                **self._field_info(field)}

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
