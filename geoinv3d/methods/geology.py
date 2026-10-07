"""Geology-constrained reference models: rock units with their properties, placed on the mesh.

For a linear problem such as gravity with a convex regularization, the starting model
does not change the answer.  What does is the reference model the smallness term pulls
towards, the bounds, and how strongly each cell is pulled.  So "a geology-constrained
starting model" is built here as all three, per active cell:

    reference   the unit's value (a density contrast relative to the Bouguer reduction
                density, a susceptibility, or a resistivity as log conductivity); the
                method's background elsewhere
    lower/upper the unit's range; the job's bounds elsewhere
    weights     a multiplier of the smallness term: the unit's ``weight`` (default 5)
                where the geology is known, 1 elsewhere

and the inversion also starts from the reference (for MT / DC, non-linear, that is the
starting model proper).  Where the data need something else, they still win: the
constraints are a hypothesis the data may accept or reject (compare the misfit with and
without them).

Units come from a table (``units``): each either lists the rock types whose samples give
its value and range (mean and min–max of the sample densities minus the background,
widened by ``margin``; for resistivity the geometric mean, and a factor), or gives
``value`` / ``lower`` / ``upper`` directly (resistivities in ohm m).  A unit marked
``sharp`` keeps little smoothing across its boundaries (``sharp_factor`` of the usual,
0.01): the model may jump there, as across ModEM's covariance "tears".

Sources place units on the mesh, in order (a later source overrides an earlier one in
the cells they share):

    samples    the cell column under each rock sample, down to ``depth_m``
    boreholes  the cells along each hole's trace (collar, azimuth, inclination, length),
               the unit from a field of the hole (e.g. its commodity) or given; or holes
               given in the spec (``holes``), each logged in depth intervals with a unit
               each (see below); with ``radius_m``, also the cells around them, in part
    body       a polygon (or box) between two depths, optionally dipping: an interpreted body
    map        polygons of a vector file (e.g. a geological map), extruded like a body
    layers     a stack of layers (thicknesses, the last one possibly down to the bottom),
               everywhere or inside a polygon/box, below the ground or by elevation
               (``reference``), its interfaces flat or tilted (``dip``, ``dip_direction``
               about ``origin``)

Bodies, maps and layers give each cell the share of its volume they cover (the footprint
sampled where an edge crosses it; depths exactly), so layers or bodies thinner than the
cells are mixed into them instead of lost: a cell's reference, bounds and weight are the
volume averages of what it holds (in log conductivity by default for resistivity, or
keeping the conductance or the transverse resistance: ``mixing``).  Samples and boreholes
mark whole cells, as the depth a sample stands for is a guess.  A hole with ``radius_m``
also reaches the cells around its trace: a cell whose centre is a distance d from the
nearest point of the trace takes the share 1 - d / radius of that point's unit (none
beyond the radius), so its reference, bounds and weight go over linearly from the log's
at the hole to what was there before at the radius.

Holes given in the spec (the model builder writes them) are logged by interval::

    {"type": "boreholes", "crs": "job", "radius_m": 0,
     "holes": [{"name": "BH1", "x": 2500, "y": 1800, "collar_m": null, "azimuth": 0,
                "inclination": 90,
                "intervals": [{"from_m": 0, "to_m": 120, "unit": "BH1 0-120"}, ...]}]}

with depths along the hole from its collar (``collar_m``, an elevation; none: on the mesh's
ground), the inclination from the horizontal (90: straight down) and the azimuth clockwise from
north.  A cell the hole passes through takes the units of the intervals in it by their length
there, so an interval boundary inside a cell mixes the two as a body's edge does.

Coordinates of samples, holes and polygons may be longitude/latitude (they are then
projected to the job's CRS) or metres in it (``"crs": "job"`` says so; otherwise it is
guessed).  Depths are below the ground surface.

A spec, as the pipeline takes it in ``params["geology"]`` (files relative to the data)::

    {"property": "density", "background": 2.67,
     "samples": {"file": "rock_samples.csv", "window_pad_m": 20000},
     "units": {"BIF": {"rock_types": ["banded iron formation"], "weight": 10},
               "mafic": {"rock_types": ["amphibolite", "dolerite*", "gabbro", "*metabasalt*"]},
               "granitoid": {"rock_types": ["*granite*", "granodiorite", "*gneiss*"],
                             "weight": 2}},
     "sources": [{"type": "samples", "depth_m": 500},
                 {"type": "boreholes", "file": "holes.shp",
                  "unit_by": {"field": "commodity_", "values": {"IRON*": "BIF"}}},
                 {"type": "body", "unit": "BIF", "polygon": [[x, y], ...],
                  "top_m": 0, "bottom_m": 1500, "dip": 70, "dip_direction": 90}],
     "unconstrained": {"lower": -0.3, "upper": 0.8}}

and a layered resistivity prior for MT / DC::

    {"property": "resistivity", "mixing": "conductance",
     "units": {"cover": {"value": 30}, "clay": {"value": 3, "sharp": true},
               "basement": {"value": 3000, "lower": 1000, "upper": 30000}},
     "sources": [{"type": "layers", "crs": "job", "top_m": 0, "dip": 5, "dip_direction": 90,
                  "layers": [{"unit": "cover", "thickness_m": 40},
                             {"unit": "clay", "thickness_m": 15},
                             {"unit": "basement"}]}]}
"""

from __future__ import annotations

import csv
import fnmatch
import os
import re
from dataclasses import dataclass, field

import numpy as np

PROPERTIES = ("density", "susceptibility", "resistivity")
# density: the Bouguer reduction density the contrasts are relative to; resistivity values
# are absolute (ohm m) and inverted as log conductivity, so it has no background here
BACKGROUND = {"density": 2.67, "susceptibility": 0.0, "resistivity": None}
# a unit's default range around its value: +- for density and susceptibility, a factor
# (x / and /) for resistivity
MARGIN = {"density": 0.05, "susceptibility": 0.005, "resistivity": 2.0}
DEFAULT_RESISTIVITY = 100.0     # ohm m, the MT / DC methods' default background (0.01 S/m)
DEFAULT_WEIGHT = 5.0


@dataclass
class Unit:
    """A rock unit: its reference value, range and weight, in model units (contrasts for
    density, SI for susceptibility, log conductivity = -ln(resistivity) for resistivity)."""

    name: str
    value: float
    lower: float
    upper: float
    weight: float = DEFAULT_WEIGHT
    rock_types: tuple = ()
    n_samples: int = 0
    source: str = "given"
    prop: str = "density"
    n_touched: int = 0          # cells it was put in (before later sources)
    volume_cells: float = 0.0   # the volume it was given, in cells (shares summed)
    sharp: bool = False         # little smoothing across its boundaries

    def as_dict(self) -> dict:
        out = {"name": self.name, "value": round(self.value, 4), "lower": round(self.lower, 4),
               "upper": round(self.upper, 4), "weight": self.weight,
               "rock_types": list(self.rock_types), "n_samples": self.n_samples,
               "source": self.source, "n_touched": self.n_touched,
               "volume_cells": self.volume_cells, "sharp": self.sharp}
        if self.prop == "resistivity":
            sig = lambda v: float(f"{v:.4g}")                                    # noqa: E731
            out.update(value_ohm_m=sig(np.exp(-self.value)), lower_ohm_m=sig(np.exp(-self.upper)),
                       upper_ohm_m=sig(np.exp(-self.lower)))
        return out


@dataclass
class GeologyConstraints:
    """Per active cell: the largest unit (-1 where the unconstrained part is larger), the
    constrained share of the cell's volume, and the reference, bounds and smallness weights."""

    units: list
    unit_index: np.ndarray
    reference: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray
    share: np.ndarray = None
    property: str = "density"
    background: float | None = 0.0
    mixing: str = "log"
    sources: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        counts = np.bincount(self.unit_index[self.unit_index >= 0], minlength=len(self.units))
        share = self.share if self.share is not None else (self.unit_index >= 0).astype(float)
        out = {"property": self.property, "background": self.background,
               "units": [dict(u.as_dict(), n_cells=int(c)) for u, c in zip(self.units, counts)],
               "n_cells": int(self.unit_index.size),
               "n_constrained": int(np.sum(self.unit_index >= 0)),
               "n_touched": int(np.sum(share > 1e-9)),
               "n_partial": int(np.sum((share > 1e-9) & (share < 1 - 1e-9))),
               "sources": self.sources, "notes": self.notes}
        if self.property == "resistivity":
            out["mixing"] = self.mixing
        return out

    def smoothness_labels(self) -> np.ndarray | None:
        """Per cell, 1 + the index of its unit where that unit is sharp, else 0 (the model
        may jump between cells of different labels), or None when no unit is sharp."""
        sharp = np.array([u.sharp for u in self.units] + [False])
        if not sharp.any():
            return None
        return np.where(sharp[self.unit_index], self.unit_index + 1, 0)


# ── Inputs ─────────────────────────────────────────────────────────────


def _matches(rock: str, patterns) -> bool:
    rock = " ".join(str(rock).lower().split())
    return any(fnmatch.fnmatch(rock, str(p).lower()) for p in patterns)


def _to_crs(x, y, crs, geographic: bool | None = None):
    """Project lon/lat to ``crs`` (when the coordinates are degrees)."""
    from ..io.crs import GEOGRAPHIC, looks_geographic, project

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if geographic is None:
        geographic = looks_geographic(x, y)
    if not geographic:
        return x, y
    if not crs:
        raise ValueError("Geology inputs are in longitude/latitude but the job has no CRS")
    return project(x, y, GEOGRAPHIC, crs)


def _unit_factor(header: str) -> float:
    """Multiplier to SI (susceptibility) from a header such as 'X 10^(-6) CGS units'."""
    h = header.lower()
    m = re.search(r"10\s*\^\s*\(?\s*(-?\d+)", h)
    factor = 10.0 ** int(m.group(1)) if m else 1.0
    if "cgs" in h:
        factor *= 4 * np.pi
    return factor


def read_samples(path, prop: str = "density", crs: str | None = None) -> list[dict]:
    """Rock samples {x, y, rock, value} from a CSV with position, rock type and property
    columns (names matched loosely: lat/lon or x/y, 'rock'/'lith', 'density', 'suscept',
    'resist' in ohm m or 'conduct' in S/m or mS/m; resistivity values come back in ohm m)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    low = [h.lower() for h in head]

    def col(*keys, exclude=()):
        for i, h in enumerate(low):
            if any(k in h for k in keys) and not any(e in h for e in exclude):
                return i
        return None
    ix, iy = col("lon", "easting"), col("lat", "northing")
    if ix is None or iy is None:
        ix, iy = col("x"), col("y")
    irock = col("rock", "lith")
    conductivity = False
    if prop == "density":
        ival, factor = col("density"), 1.0
    elif prop == "susceptibility":
        ival = col("suscept")
        factor = _unit_factor(head[ival]) if ival is not None else 1.0
    else:
        ival, factor = col("resist"), 1.0
        if ival is None:
            ival, conductivity = col("conduct"), True
            factor = 1e-3 if ival is not None and "ms/m" in low[ival] else 1.0
    if None in (ix, iy, irock, ival):
        raise ValueError(f"'{os.path.basename(path)}' needs position, rock type and {prop} columns; "
                         f"found {head}")
    out = []
    for r in rows[1:]:
        try:
            x, y, v = float(r[ix]), float(r[iy]), float(r[ival]) * factor
        except (ValueError, IndexError):
            continue
        if prop == "resistivity":
            if not v > 0:
                continue
            v = 1.0 / v if conductivity else v
        out.append({"x": x, "y": y, "rock": r[irock].strip(), "value": v})
    if not out:
        raise ValueError(f"No usable samples in '{os.path.basename(path)}'")
    xs, ys = _to_crs([s["x"] for s in out], [s["y"] for s in out], crs)
    for s, x, y in zip(out, xs, ys):
        s["x"], s["y"] = float(x), float(y)
    return out


def units_from_samples(units_spec: dict, samples: list | None, prop: str = "density",
                       background: float | None = None, skipped: list | None = None) -> list[Unit]:
    """The unit table: values from matching samples, or as given (see the module docstring).

    A unit whose samples are missing raises, unless ``skipped`` is a list: the unit is
    then left out and its name appended there.
    """
    if prop == "resistivity":
        # values, ranges and samples in ohm m; the model is log conductivity = -ln(rho),
        # so a unit's lowest resistivity is its highest model value; samples average
        # geometrically; the margin is a factor
        base = 0.0
        to_model = lambda v: -np.log(float(v))                                   # noqa: E731
    else:
        bg = BACKGROUND[prop] if background is None else float(background)
        base = bg if prop == "density" else 0.0          # contrasts for density only
        to_model = lambda v: float(v) - base                                      # noqa: E731
    out = []
    for name, spec in units_spec.items():
        spec = dict(spec or {})
        weight = float(spec.get("weight", DEFAULT_WEIGHT))
        margin = float(spec.get("margin", MARGIN[prop]))
        if prop == "resistivity":
            if not margin >= 1:
                raise ValueError(f"Unit '{name}': a resistivity margin is a factor >= 1")
            margin = np.log(margin)
        rocks = tuple(spec.get("rock_types") or ())
        # given bounds, in model units (swapped for resistivity)
        lo_given = spec.get("upper" if prop == "resistivity" else "lower")
        hi_given = spec.get("lower" if prop == "resistivity" else "upper")
        if "value" in spec:
            if prop == "resistivity":
                v = to_model(spec["value"])
                lo = v - margin if lo_given is None else to_model(lo_given)
                hi = v + margin if hi_given is None else to_model(hi_given)
            else:     # given values of density are contrasts already
                v = float(spec["value"])
                lo = v - margin if lo_given is None else float(lo_given)
                hi = v + margin if hi_given is None else float(hi_given)
            out.append(Unit(name, v, lo, hi, weight, rocks, 0, "given", prop,
                            sharp=bool(spec.get("sharp", False))))
            continue
        if not rocks:
            raise ValueError(f"Unit '{name}' needs 'value' or 'rock_types'")
        vals = np.array([to_model(s["value"]) for s in (samples or []) if _matches(s["rock"], rocks)])
        if vals.size < int(spec.get("min_samples", 1)):
            if skipped is not None:
                skipped.append(name)
                continue
            raise ValueError(f"Unit '{name}': no samples of {list(rocks)}; give its 'value'")
        v = float(vals.mean())
        if prop == "resistivity":
            lo = to_model(lo_given) if lo_given is not None else float(vals.min() - margin)
            hi = to_model(hi_given) if hi_given is not None else float(vals.max() + margin)
        else:
            lo = float(lo_given) if lo_given is not None else float(vals.min() - margin)
            hi = float(hi_given) if hi_given is not None else float(vals.max() + margin)
        out.append(Unit(name, v, min(lo, v), max(hi, v), weight, rocks, int(vals.size), "samples",
                        prop, sharp=bool(spec.get("sharp", False))))
    return out


# ── Geometry ───────────────────────────────────────────────────────────


def points_in_polygon(px, py, rings) -> np.ndarray:
    """Even-odd rule over all rings (holes included), vectorized over the points."""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        r = np.asarray(ring, dtype=float)
        if len(r) < 3:
            continue
        x1, y1 = r[:, 0], r[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        for a, b, c, d in zip(x1, y1, x2, y2):
            if b == d:
                continue
            crosses = (b > py) != (d > py)
            xint = (c - a) * (py - b) / (d - b) + a
            inside ^= crosses & (px < xint)
    return inside


def _near_edges(px, py, radius, rings) -> np.ndarray:
    """Points closer than ``radius`` (per point) to any edge of the rings."""
    near = np.zeros(px.shape, dtype=bool)
    r2 = np.asarray(radius, dtype=float) ** 2
    for ring in rings:
        r = np.asarray(ring, dtype=float)
        if len(r) < 2:
            continue
        for (ax, ay), (bx, by) in zip(r, np.roll(r, -1, axis=0)):
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / ll, 0.0, 1.0) if ll > 0 else 0.0
            near |= (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2 < r2
    return near


def _bbox(rings):
    pts = np.concatenate([np.asarray(r, dtype=float)[:, :2] for r in rings if len(r)])
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def _overlap(a0, a1, b0, b1):
    """Length of the overlap of the intervals [a0, a1] and [b0, b1]."""
    return np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)


class _Cells:
    """The active cells: centres, half sizes, depths below the ground and each cell's
    height below the ground (a cell cut by the ground counts only its part below it)."""

    def __init__(self, centres, half, surface, subsample: int = 4):
        self.x, self.y, self.z = centres[:, 0], centres[:, 1], centres[:, 2]
        self.hx, self.hy, self.hz = half[:, 0], half[:, 1], half[:, 2]
        self.n = self.x.size
        self.surface = surface
        self.ground = np.asarray(surface(self.x, self.y), dtype=float)
        self.depth = self.ground - self.z
        # the cell's extent below the ground, in depth and in elevation
        self.dtop = np.maximum(self.ground - (self.z + self.hz), 0.0)
        self.dbot = self.ground - (self.z - self.hz)
        self.ebot = self.z - self.hz
        self.etop = np.minimum(self.z + self.hz, self.ground)
        self.height = np.clip(self.dbot - self.dtop, 0.0, None)
        self.sub = max(int(subsample), 1)

    def containing(self, px, py, pz) -> np.ndarray:
        """Index of the cell containing each point (-1 outside every cell)."""
        out = np.full(len(px), -1)
        for k, (a, b, c) in enumerate(zip(px, py, pz)):
            hit = np.flatnonzero((np.abs(self.x - a) <= self.hx) & (np.abs(self.y - b) <= self.hy)
                                 & (np.abs(self.z - c) <= self.hz))
            if hit.size:
                out[k] = hit[0]
        return out

    def columns(self, px, py, radius: float = 0.0) -> np.ndarray:
        """Cells whose footprint lies within ``radius`` of any of the points."""
        hit = np.zeros(self.x.shape, dtype=bool)
        for a, b in zip(px, py):
            hit |= (np.abs(self.x - a) <= self.hx + radius) & (np.abs(self.y - b) <= self.hy + radius)
        return hit

    def in_bbox(self, box, pad_x=0.0, pad_y=0.0) -> np.ndarray:
        """Cells whose footprint meets the box [x0, x1, y0, y1] widened by the pads."""
        x0, x1, y0, y1 = box
        return ((self.x + self.hx >= x0 - pad_x) & (self.x - self.hx <= x1 + pad_x)
                & (self.y + self.hy >= y0 - pad_y) & (self.y - self.hy <= y1 + pad_y))

    def offsets(self):
        """Sub-point offsets in units of the half sizes: an n x n grid over the footprint."""
        o = (np.arange(self.sub) + 0.5) / self.sub * 2.0 - 1.0
        ox, oy = np.meshgrid(o, o, indexing="ij")
        return ox.ravel(), oy.ravel()

    def inside_share(self, k, rings, dx=0.0, dy=0.0) -> np.ndarray:
        """Share of the footprints of cells ``k`` inside the rings (moved by -dx, -dy):
        the centre test, refined on an n x n grid of points where an edge is near."""
        x, y = self.x[k] - dx, self.y[k] - dy
        hx, hy = self.hx[k], self.hy[k]
        share = points_in_polygon(x, y, rings).astype(float)
        near = _near_edges(x, y, np.hypot(hx, hy), rings)
        if self.sub > 1 and near.any():
            j = np.flatnonzero(near)
            ox, oy = self.offsets()
            px = (x[j, None] + hx[j, None] * ox[None, :]).ravel()
            py = (y[j, None] + hy[j, None] * oy[None, :]).ravel()
            share[j] = points_in_polygon(px, py, rings).reshape(j.size, -1).mean(axis=1)
        return share

    def vertical_share(self, k, top, bottom, elevations: bool) -> np.ndarray:
        """Share of the below-ground height of cells ``k`` between ``top`` and ``bottom``
        (depths below the ground, or elevations with top above bottom)."""
        if elevations:
            ov = _overlap(self.ebot[k], self.etop[k], bottom, top)
        else:
            ov = _overlap(self.dtop[k], self.dbot[k], top, bottom)
        h = self.height[k]
        return np.where(h > 0, ov / np.where(h > 0, h, 1.0), 0.0)


def _body_share(cells: _Cells, rings, top: float, bottom: float, dip: float = 90.0,
                dip_direction: float = 0.0, elevations: bool = False, slices: int = 6):
    """(cell indices, shares) of a polygon extruded between two depths (or elevations),
    its outline moving down-dip with depth: the share of each cell's volume inside it."""
    span = abs(bottom - top)
    shift = span / np.tan(np.radians(dip)) if dip < 90.0 else 0.0
    sx, sy = shift * np.sin(np.radians(dip_direction)), shift * np.cos(np.radians(dip_direction))
    x0, x1, y0, y1 = _bbox(rings)
    box = (x0 + min(sx, 0.0), x1 + max(sx, 0.0), y0 + min(sy, 0.0), y1 + max(sy, 0.0))
    k = np.flatnonzero(cells.in_bbox(box))
    if not k.size:
        return k, np.zeros(0)
    vs = cells.vertical_share(k, top, bottom, elevations)
    keep = vs > 0
    k, vs = k[keep], vs[keep]
    if not k.size:
        return k, vs
    if dip >= 90.0:
        return k, vs * cells.inside_share(k, rings)
    # the part of each cell inside the body's depth range, in slices; each slice's outline
    # lies (d - top) / tan(dip) down-dip of the top one, d the slice's middle below the top
    if elevations:
        a, b = np.maximum(cells.ebot[k], bottom), np.minimum(cells.etop[k], top)
        below_top = lambda t: top - (a + (b - a) * t)                               # noqa: E731
    else:
        a, b = np.maximum(cells.dtop[k], top), np.minimum(cells.dbot[k], bottom)
        below_top = lambda t: a + (b - a) * t - top                                   # noqa: E731
    share = np.zeros(k.size)
    for j in range(slices):
        d = np.clip(below_top((j + 0.5) / slices), 0.0, None) / np.tan(np.radians(dip))
        share += cells.inside_share(k, rings, d * np.sin(np.radians(dip_direction)),
                                    d * np.cos(np.radians(dip_direction)))
    return k, vs * share / slices


def _layer_shares(cells: _Cells, rings, top: float, thicknesses, dip: float = 0.0,
                  dip_direction: float = 0.0, origin=(0.0, 0.0), elevations: bool = False):
    """Per layer of a stack, (cell indices, shares): the share of each cell's volume in it.

    Interfaces are planes: the stack's top at ``top`` (depth below the ground, or
    elevation) at ``origin``, sinking by tan(dip) per metre towards ``dip_direction``;
    thicknesses are vertical, the last one may be None (down to the bottom of the mesh).
    """
    if rings:
        k = np.flatnonzero(cells.in_bbox(_bbox(rings)))
        outline = cells.inside_share(k, rings) if k.size else np.zeros(0)
        keep = outline > 0
        k, outline = k[keep], outline[keep]
    else:
        k, outline = np.arange(cells.n), np.ones(cells.n)
    slope = np.tan(np.radians(dip)) if dip > 0 else 0.0
    ux, uy = np.sin(np.radians(dip_direction)), np.cos(np.radians(dip_direction))
    # how much deeper the interfaces lie at each cell centre, and their spread over the footprint
    tilt = slope * ((cells.x[k] - origin[0]) * ux + (cells.y[k] - origin[1]) * uy)
    spread = slope * (np.abs(ux) * cells.hx[k] + np.abs(uy) * cells.hy[k])
    edges = np.concatenate([[0.0], np.cumsum([t if t is not None else np.inf for t in thicknesses])])
    if elevations:   # work in "metres below the stack's top" = top - elevation
        c0, c1 = top - cells.etop[k], top - cells.ebot[k]
    else:
        c0, c1 = cells.dtop[k] - top, cells.dbot[k] - top
    h = np.where(c1 > c0, c1 - c0, 1.0)
    ox, oy = cells.offsets()
    out = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        lo, hi = e0 + tilt, e1 + tilt
        vs = np.where(c1 > c0, _overlap(c0, c1, lo, hi) / h, 0.0)
        # where a tilted interface crosses the cell, average over points of its footprint
        cut = (spread > 0) & (((lo - spread < c1) & (lo + spread > c0))
                              | (np.isfinite(hi) & (hi - spread < c1) & (hi + spread > c0)))
        if cut.any():
            j = np.flatnonzero(cut)
            dt = slope * (ox[None, :] * cells.hx[k][j, None] * ux + oy[None, :] * cells.hy[k][j, None] * uy)
            ov = _overlap(c0[j, None], c1[j, None], lo[j, None] + dt, hi[j, None] + dt)
            vs[j] = np.where(c1[j] > c0[j], ov.mean(axis=1) / h[j], 0.0)
        share = vs * outline
        on = share > 1e-9
        out.append((k[on], share[on]))
    return out


def _polygon_rings(spec, crs):
    """Rings of a body: ``polygon`` [[x, y], ...] or ``box`` [west, east, south, north]."""
    if "box" in spec:
        w, e, s, n = (float(v) for v in spec["box"])
        ring = [(w, s), (e, s), (e, n), (w, n)]
    elif "polygon" in spec:
        ring = [tuple(map(float, p[:2])) for p in spec["polygon"]]
    else:
        raise ValueError("A body needs a 'polygon' or a 'box'")
    x, y = _to_crs([p[0] for p in ring], [p[1] for p in ring], crs, _geographic(spec))
    return [list(zip(x, y))]


def _geographic(spec) -> bool | None:
    """A source's ``crs``: "EPSG:4326" / "WGS84" = degrees, anything else (e.g. "job", as
    the upload page sends) = metres in the job's CRS, none = guessed from the numbers."""
    c = str(spec.get("crs", "") or "").upper()
    return True if c in ("EPSG:4326", "WGS84") else (False if c else None)


def _unit_of(props: dict, spec: dict, names: set) -> str | None:
    """The unit of a feature: ``unit`` of the source, or ``unit_by`` {field, values: {pattern: unit}}."""
    if spec.get("unit"):
        return spec["unit"]
    by = spec.get("unit_by") or {}
    value = props.get(by.get("field"))
    if value is None:
        return None
    for pattern, unit in (by.get("values") or {}).items():
        if fnmatch.fnmatch(str(value).lower(), str(pattern).lower()):
            return unit
    return str(value) if str(value) in names else None


def _borehole_cells(cells: _Cells, spec: dict, data_dir: str, crs, names: set, step: float):
    """(cell index, unit) pairs along the holes of a shapefile / GeoJSON / CSV, the number
    of holes used and read, and the traces: (n, 3) points with their unit, per hole."""
    from ..io.vector import read_vector

    path = os.path.join(data_dir, spec["file"])
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            feats = [{"type": "Point", "properties": r} for r in csv.DictReader(fh)]
        fx, fy = spec.get("x_field", "x"), spec.get("y_field", "y")
        for f in feats:
            f["coordinates"] = (float(f["properties"][fx]), float(f["properties"][fy]))
        geographic = None
    else:
        feats, geographic = read_vector(path)
    feats = [f for f in feats if f["type"] == "Point"]
    xs, ys = _to_crs([f["coordinates"][0] for f in feats], [f["coordinates"][1] for f in feats],
                     crs, geographic)
    az_f = spec.get("azimuth_field", "bearing")
    inc_f = spec.get("inclination_field", "cl_inclina")
    len_f = spec.get("length_field", "length_m")
    extend = float(spec.get("extend_m", 0.0))
    pairs, used, traces = [], 0, []
    for f, x, y in zip(feats, xs, ys):
        unit = _unit_of(f["properties"], spec, names)
        if unit is None:
            continue

        def num(key, default):
            try:
                v = float(f["properties"].get(key))
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default
        length = num(len_f, 0.0) + extend
        inc = num(inc_f, 90.0)
        inc = 90.0 if inc <= 0 else inc       # 0 / missing: taken as vertical
        az = np.radians(num(az_f, 0.0))
        s = np.minimum(np.arange(0.0, max(length, 0.0) + step, step), length)
        dx, dy = np.cos(np.radians(inc)) * np.sin(az), np.cos(np.radians(inc)) * np.cos(az)
        dz = np.sin(np.radians(inc))
        px, py = x + s * dx, y + s * dy
        # the collar on the mesh's ground (default), or at the hole's own elevation
        if spec.get("collar", "surface") == "surface":
            top = float(np.asarray(cells.surface(np.array([x]), np.array([y]))).ravel()[0])
        else:
            top = num(spec.get("collar_field", "rl_collar_"), np.nan)
        if not np.isfinite(top):
            continue
        pz = top - s * dz - 1e-6
        idx = cells.containing(px, py, pz)
        traces.append((np.column_stack([px, py, pz]), unit))
        idx = idx[idx >= 0]
        if idx.size:
            used += 1
            pairs += [(int(i), unit) for i in np.unique(idx)]
    return pairs, used, len(feats), traces


def _logged_hole_cells(cells: _Cells, spec: dict, crs, names: set, step: float):
    """Holes given in the spec, logged by interval (see the module docstring): one (cell, unit)
    pair per point sampled every ``step`` along each interval (so a cell holds its units in
    proportion to their length in it), the holes used and given, and the traces per interval."""
    holes = [h for h in spec.get("holes") or [] if isinstance(h, dict)]
    if not holes:
        return [], 0, 0, []
    try:
        hx, hy = [float(h["x"]) for h in holes], [float(h["y"]) for h in holes]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Every hole needs numbers x and y")
    xs, ys = _to_crs(hx, hy, crs, _geographic(spec))
    pairs, used, traces = [], 0, []
    for h, x, y in zip(holes, xs, ys):
        inc = float(h.get("inclination") or 90.0)
        inc = 90.0 if inc <= 0 or inc > 90 else inc
        az = np.radians(float(h.get("azimuth") or 0.0))
        dx, dy = np.cos(np.radians(inc)) * np.sin(az), np.cos(np.radians(inc)) * np.cos(az)
        dz = np.sin(np.radians(inc))
        collar = h.get("collar_m")
        top = (float(np.asarray(cells.surface(np.array([x]), np.array([y]))).ravel()[0])
               if collar is None or collar == "" else float(collar))
        hit = False
        for iv in h.get("intervals") or []:
            unit = iv.get("unit")
            if unit is None or unit not in names:
                continue
            a, b = float(iv["from_m"]), float(iv["to_m"])
            if not b > a:
                raise ValueError(f"Hole {h.get('name', '?')}: an interval from {a:g} m to {b:g} m")
            # points at the middle of steps of at most a quarter of `step`: each stands for the
            # same length, so a cell's units are shared by their length in it to within a few %
            n = max(int(np.ceil(4 * (b - a) / step)), 1)
            s = a + (np.arange(n) + 0.5) * (b - a) / n
            px, py, pz = x + s * dx, y + s * dy, top - s * dz
            idx = cells.containing(px, py, pz)
            traces.append((np.column_stack([px, py, pz]), unit))
            ok = idx >= 0
            if ok.any():
                hit = True
                pairs += [(int(i), unit) for i in idx[ok]]
        used += int(hit)
    return pairs, used, len(holes), traces


def _by_length(pairs, names: dict):
    """[(unit index, cells, shares)] from (cell, unit) pairs, one per equal length of hole: each
    cell filled by its units in proportion to their length in it."""
    from collections import Counter
    count = Counter(pairs)
    total = Counter(i for i, _ in pairs)
    out = {}
    for (i, u), c in count.items():
        out.setdefault(names[u], ([], []))
        out[names[u]][0].append(i)
        out[names[u]][1].append(c / total[i])
    return [(k, np.array(i), np.array(f)) for k, (i, f) in out.items()]


def _borehole_halo(cells: _Cells, traces, pairs, radius: float, names: dict):
    """(unit index, cells, shares) around the holes: 1 - d / radius of the unit of the
    nearest trace point (d from the cell's centre), 1 in the cells the traces pass through."""
    from scipy.spatial import cKDTree

    pts = np.vstack([t for t, _ in traces])
    unit_of = np.concatenate([np.full(len(t), names[u]) for t, u in traces])
    centres = np.column_stack([cells.x, cells.y, cells.z])
    d, j = cKDTree(pts).query(centres, distance_upper_bound=radius)
    near = np.isfinite(d)
    share = np.zeros(cells.n)
    unit = np.full(cells.n, -1)
    share[near] = 1.0 - d[near] / radius
    unit[near] = unit_of[j[near]]
    for i, u in pairs:          # the cells a trace passes through: whole, their unit as logged
        share[i], unit[i] = 1.0, names[u]
    keep = share > 0
    return [(int(k), np.flatnonzero(keep & (unit == k)), share[keep & (unit == k)])
            for k in np.unique(unit[keep])]


# ── Mixing units into cells ────────────────────────────────────────────


def _mixing(prop: str, law: str | None):
    """(to, back): the space in which a cell's contents average linearly by volume.

    Density and susceptibility average as they are (exact for gravity: the mass is kept).
    Resistivity models are log conductivity; its cells average ``log`` (the geometric mean,
    default), ``conductance`` (arithmetic mean of conductivity: a thin conductor keeps its
    conductance, as horizontal currents such as MT's see it) or ``resistance`` (arithmetic
    mean of resistivity: a thin resistor keeps its transverse resistance, as vertical DC
    currents see it).
    """
    if prop != "resistivity" or law in (None, "log"):
        return (lambda m: m), (lambda g: g)
    if law == "conductance":
        return np.exp, np.log
    if law == "resistance":
        return (lambda m: np.exp(-m)), (lambda g: -np.log(g))
    raise ValueError(f"Unknown mixing '{law}' (expected log, conductance or resistance)")


class _Mix:
    """Per cell, what the sources have put there so far: volume-weighted mixes of the units'
    reference, bounds and weights (in the mixing space), the constrained share of the cell
    and its largest unit.  A source takes its share of every cell from what was there, so a
    later source overrides an earlier one in proportion to the volume it covers."""

    def __init__(self, n, to, ref, lo, hi, weight):
        self.to = to
        with np.errstate(over="ignore"):
            self.ref, self.lo, self.hi = (np.full(n, to(float(v))) for v in (ref, lo, hi))
        self.w = np.full(n, float(weight))
        self.share = np.zeros(n)            # the constrained share of each cell
        self.idx = np.full(n, -1)           # its largest unit
        self.idx_share = np.zeros(n)

    def add(self, parts, units) -> np.ndarray:
        """Place ``parts`` [(unit index, cell indices, shares)], disjoint within the source.
        Returns the cells touched."""
        parts = [p for p in parts if len(p[1])]
        if not parts:
            return np.zeros(0, dtype=int)
        t = np.unique(np.concatenate([p[1] for p in parts]))
        f_sum, r, lo, hi, w = (np.zeros(t.size) for _ in range(5))
        best_k, best_f = np.full(t.size, -1), np.zeros(t.size)
        for k, i, f in parts:
            j = np.searchsorted(t, i)
            u = units[k]
            np.add.at(f_sum, j, f)
            with np.errstate(over="ignore"):
                np.add.at(r, j, f * self.to(u.value))
                np.add.at(lo, j, f * self.to(u.lower))
                np.add.at(hi, j, f * self.to(u.upper))
            np.add.at(w, j, f * u.weight)
            better = f > best_f[j]
            best_k[j[better]], best_f[j[better]] = k, f[better]
        over = f_sum > 1.0            # overlapping features of one source: scaled to fill the cell
        if over.any():
            s = 1.0 / f_sum[over]
            for a in (r, lo, hi, w, best_f):
                a[over] *= s
            f_sum[over] = 1.0
        keep = 1.0 - f_sum
        with np.errstate(invalid="ignore"):
            for name, add in (("ref", r), ("lo", lo), ("hi", hi), ("w", w)):
                old = getattr(self, name)
                old[t] = np.where(keep > 0, keep * old[t], 0.0) + add
        same = self.idx[t] == best_k
        self.idx_share[t] = self.idx_share[t] * keep + np.where(same, best_f, 0.0)
        take = ~same & (best_f > self.idx_share[t])
        self.idx[t[take]], self.idx_share[t[take]] = best_k[take], best_f[take]
        self.share[t] = self.share[t] * keep + f_sum
        return t


# ── Building the constraints ───────────────────────────────────────────


def build_constraints(spec: dict, centres, half_sizes, surface, data_dir: str = ".",
                      crs: str | None = None, default_bounds=(None, None),
                      default_reference: float | None = None) -> GeologyConstraints:
    """Reference model, bounds and smallness weights on the active cells.

    Args:
        spec: see the module docstring.
        centres, half_sizes: (n, 3) arrays of the active cells (m).
        surface: ``surface(x, y) -> z`` of the ground.
        data_dir: where the spec's files are.
        crs: the job's CRS, for longitude/latitude inputs.
        default_bounds: the job's (lower, upper) for cells no unit covers, in model units.
        default_reference: the model value of cells no unit covers (the method's background:
            0 for density and susceptibility, log sigma_background for resistivity).
    """
    prop = spec.get("property", "density")
    if prop not in PROPERTIES:
        raise ValueError(f"Unknown property '{prop}' (expected {list(PROPERTIES)})")
    background = spec.get("background", BACKGROUND[prop])
    background = None if background is None else float(background)
    notes = []
    # Files are optional: a missing one leaves out what depends on it, with a note
    samples = None
    sspec = spec.get("samples") or {}
    if sspec.get("file"):
        path = os.path.join(data_dir, sspec["file"])
        if os.path.exists(path):
            samples = read_samples(path, prop, crs)
        else:
            notes.append(f"Samples file '{sspec['file']}' not found: units valued by samples "
                         "and the samples source are left out")
    cells = _Cells(np.asarray(centres, dtype=float), np.asarray(half_sizes, dtype=float), surface,
                   int(spec.get("subsample", 4)))
    if samples is not None and sspec.get("window_pad_m") is not None:
        pad = float(sspec["window_pad_m"])
        x0, x1 = cells.x.min() - pad, cells.x.max() + pad
        y0, y1 = cells.y.min() - pad, cells.y.max() + pad
        samples = [s for s in samples if x0 <= s["x"] <= x1 and y0 <= s["y"] <= y1]
    dropped = []
    units = units_from_samples(spec.get("units") or {}, samples, prop, background, skipped=dropped)
    if dropped:
        notes.append(f"Units without samples left out: {dropped}")
    if not units:
        raise ValueError("The geology spec defines no usable units")
    names = {u.name: k for k, u in enumerate(units)}

    # cells no unit covers: the unconstrained value and bounds (resistivities in ohm m)
    free = spec.get("unconstrained") or {}
    ref_d = default_reference if default_reference is not None \
        else (-np.log(DEFAULT_RESISTIVITY) if prop == "resistivity" else 0.0)
    lo_d, hi_d = default_bounds
    if prop == "resistivity":
        if free.get("value") is not None:
            ref_d = -np.log(float(free["value"]))
        if free.get("upper") is not None:
            lo_d = -np.log(float(free["upper"]))
        if free.get("lower") is not None:
            hi_d = -np.log(float(free["lower"]))
    else:
        ref_d = float(free.get("value", ref_d))
        lo_d, hi_d = free.get("lower", lo_d), free.get("upper", hi_d)
    lo_d = -np.inf if lo_d is None else float(lo_d)
    hi_d = np.inf if hi_d is None else float(hi_d)
    ref_d = float(np.clip(ref_d, lo_d, hi_d))
    to, back = _mixing(prop, spec.get("mixing"))
    mix = _Mix(cells.n, to, ref_d, lo_d, hi_d, float(free.get("weight", 1.0)))

    report, thin = [], []
    touched = {k: 0 for k in range(len(units))}
    volume = {k: 0.0 for k in range(len(units))}
    step = max(float(np.min(cells.hz)), 1.0)   # half the thinnest cell: holes miss none

    def known(unit, what) -> bool:
        """A unit of the table; a left-out one is skipped, an unknown one is an error."""
        if unit in names:
            return True
        if unit in dropped:
            return False
        raise ValueError(f"{what} unit '{unit}' is not in the unit table")

    def whole(pairs):
        """(unit index, cells, shares of 1) from (cell, unit name) pairs; a later pair wins."""
        last = {i: names[unit] for i, unit in pairs}
        groups = {}
        for i, k in last.items():
            groups.setdefault(k, []).append(i)
        return [(k, np.array(sorted(v)), np.ones(len(v))) for k, v in groups.items()]

    def check_thin(label, thickness, idx):
        """A layer or body thinner than the cells it falls in: it is mixed into them."""
        if idx.size and thickness is not None:
            cell = float(np.min(cells.height[idx][cells.height[idx] > 0], initial=np.inf))
            if thickness < 0.999 * cell:
                thin.append(f"{label} ({thickness:g} m in {cell:g} m cells)")

    for source in spec.get("sources") or []:
        kind = source.get("type")
        info = {"type": kind}
        if source.get("file") and not os.path.exists(os.path.join(data_dir, source["file"])):
            notes.append(f"The {kind} source's file '{source['file']}' was not found: left out")
            report.append({**info, "file": source["file"], "skipped": "file not found", "n_cells": 0})
            continue
        parts = []
        if kind == "samples":
            if samples is None:
                if not sspec.get("file"):
                    raise ValueError("A 'samples' source needs 'samples': {'file': ...}")
                report.append({**info, "skipped": "no samples", "n_cells": 0})
                continue
            depth = float(source.get("depth_m", 500.0))
            radius = float(source.get("radius_m", 0.0))
            n_used, pairs = 0, []
            for s in samples:
                unit = next((u.name for u in units if u.rock_types and _matches(s["rock"], u.rock_types)),
                            None)
                if unit is None:
                    continue
                # every cell of the column that overlaps 0..depth below the ground (the
                # first cell under the ground always, however thick): whole cells, as the
                # depth a sample stands for is a guess
                hit = cells.columns([s["x"]], [s["y"]], radius) & (cells.depth + cells.hz > 0) \
                    & (cells.depth - cells.hz < depth)
                if hit.any():
                    pairs += [(int(i), unit) for i in np.flatnonzero(hit)]
                    n_used += 1
            parts = whole(pairs)
            info.update(n_features=n_used, depth_m=depth)
        elif kind == "boreholes":
            logged = source.get("holes") is not None
            find = _logged_hole_cells if logged else _borehole_cells
            args = (cells, source, crs, set(names) | set(dropped), step) if logged else \
                (cells, source, data_dir, crs, set(names) | set(dropped), step)
            pairs, n_used, n_total, traces = find(*args)
            pairs = [(i, unit) for i, unit in pairs if known(unit, "Borehole")]
            radius = float(source.get("radius_m", 0.0) or 0.0)
            if radius > 0:
                if logged:          # a cell the hole passes through: its longest interval
                    from collections import Counter
                    best = {}
                    for (i, u), c in Counter(pairs).items():
                        if c > best.get(i, (None, 0))[1]:
                            best[i] = (u, c)
                    last = {i: u for i, (u, _) in best.items()}
                else:
                    last = dict(pairs)          # a later hole wins a cell, as without a radius
                parts = _borehole_halo(cells, [(t, u) for t, u in traces if u in names],
                                       list(last.items()), radius, names)
            else:
                parts = _by_length(pairs, names) if logged else whole(pairs)
            info.update(file=source.get("file"), n_features=n_used, n_holes=n_total)
            if radius > 0:
                info["radius_m"] = radius
        elif kind in ("body", "map"):
            if kind == "body":
                unit = source.get("unit")
                if not known(unit, "Body"):
                    report.append({**info, "unit": unit, "skipped": "unit left out", "n_cells": 0})
                    continue
                features = [(_polygon_rings(source, crs), unit)]
            else:
                from ..io.vector import read_vector
                feats, geographic = read_vector(os.path.join(data_dir, source["file"]))
                features = []
                for f in feats:
                    if f["type"] != "Polygon":
                        continue
                    unit = _unit_of(f["properties"], source, set(names))
                    if unit is None or unit not in names:
                        continue
                    rings = []
                    for ring in f["coordinates"]:
                        rx, ry = _to_crs([p[0] for p in ring], [p[1] for p in ring], crs, geographic)
                        rings.append(list(zip(rx, ry)))
                    features.append((rings, unit))
                info["file"] = source.get("file")
            elev = bool(source.get("elevations", False)) or source.get("reference") == "elevation"
            top, bottom = float(source.get("top_m", 0.0)), float(source.get("bottom_m", 1000.0))
            for rings, unit in features:
                idx, share = _body_share(cells, rings, top, bottom, float(source.get("dip", 90.0)),
                                         float(source.get("dip_direction", 0.0)), elev)
                on = share > 1e-9
                parts.append((names[unit], idx[on], share[on]))
                if kind == "body":
                    check_thin(f"body {unit}", abs(bottom - top), idx[on])
            info.update(n_features=len(features), top_m=top, bottom_m=bottom)
        elif kind == "layers":
            rings = _polygon_rings(source, crs) if ("polygon" in source or "box" in source) else None
            elev = source.get("reference") == "elevation" or bool(source.get("elevations", False))
            layers = source.get("layers") or []
            thick = [lay.get("thickness_m") for lay in layers]
            if any(t is None for t in thick[:-1]):
                raise ValueError("Only the last layer of a stack may go without 'thickness_m'")
            thick = [None if t is None else float(t) for t in thick]
            if any(t is not None and t <= 0 for t in thick):
                raise ValueError("Layer thicknesses must be positive")
            origin = source.get("origin")
            if origin is None:
                if rings:
                    x0, x1, y0, y1 = _bbox(rings)
                    origin = ((x0 + x1) / 2, (y0 + y1) / 2)
                else:
                    origin = (float(np.mean(cells.x)), float(np.mean(cells.y)))
            else:
                ox, oy = _to_crs([float(origin[0])], [float(origin[1])], crs, _geographic(source))
                origin = (float(ox[0]), float(oy[0]))
            top = float(source.get("top_m", 0.0))
            shares = _layer_shares(cells, rings, top, thick, float(source.get("dip", 0.0)),
                                   float(source.get("dip_direction", 0.0)), origin, elev)
            for lay, t, (idx, share) in zip(layers, thick, shares):
                # a layer without a unit is left free (its thickness still counts)
                if lay.get("unit") is not None and known(lay["unit"], "Layer"):
                    parts.append((names[lay["unit"]], idx, share))
                    check_thin(f"layer {lay['unit']}", t, idx)
            info.update(n_layers=len(layers), top_m=top,
                        reference="elevation" if elev else "depth")
        else:
            raise ValueError(f"Unknown geology source type '{kind}' "
                             "(expected samples, boreholes, body, map or layers)")
        for k, i, f in parts:
            touched[k] += int(i.size)
            volume[k] += float(np.sum(f))
        hit = mix.add(parts, units)
        info["n_cells"] = int(hit.size)
        by_unit = {}
        for k, i, f in parts:
            if i.size:
                by_unit[units[k].name] = by_unit.get(units[k].name, 0) + int(i.size)
        info["cells_by_unit"] = by_unit
        if not hit.size:
            notes.append(f"The {kind} source put no unit on any cell (outside the mesh?)")
        report.append(info)

    if thin:
        notes.append("Thinner than the cells they fall in, so mixed by volume into them: "
                     + ", ".join(thin) + ". Thinner cells would resolve them")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        reference, lower, upper = back(mix.ref), back(mix.lo), back(mix.hi)
    reference = np.clip(reference, lower, upper)
    # a cell's unit: its largest one, where that outweighs the unconstrained part
    idx = np.where(mix.idx_share >= 1.0 - mix.share - 1e-9, mix.idx, -1)
    wide = [u.name for u in units if u.upper > hi_d or u.lower < lo_d]
    if wide and (np.isfinite(lo_d) or np.isfinite(hi_d)):
        notes.append(f"Units {wide} reach beyond the unconstrained bounds: their cells use the "
                     "units' own range")
    for k, u in enumerate(units):
        u.n_touched, u.volume_cells = touched[k], round(volume[k], 3)
    return GeologyConstraints(units=units, unit_index=idx, reference=reference, lower=lower,
                              upper=upper, weights=mix.w, share=mix.share, property=prop,
                              background=background, mixing=spec.get("mixing") or "log",
                              sources=report, notes=notes)
