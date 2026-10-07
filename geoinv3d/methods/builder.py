"""Models of the model builder (``geoinv3d/viz/model_builder.html``, ``builder`` version 2) in
Python: as the geology spec of one property, and as that property's values on a mesh.

The spec is the one the upload page's model step submits (``mbSpec`` / ``mbUnit`` in
``geoinv3d/viz/dag_interactive.html``: keep the two in step): every body, layer and logged
borehole interval with a value of the property is a unit, its range the value +- a margin
(x / and / 2 for resistivity) unless given, ``fixed`` narrowing it ten-fold with a weight of at
least 100; densities go as contrasts to the model's background density (``free.density.value``,
2.67 g/cc by default).

The values on a mesh are the spec's reference model (``geology.build_constraints``): each cell
the volume average of what it holds, the rest of the ground ``free``'s value — a "true model"
for synthetic data from a model drawn on the page.

A builder (see the page for every field)::

    {"version": 2, "ground": 0,
     "items": [{"kind": "body", "name": "ore", "shape": "box", "box": [w, e, s, n],
                "ref": "depth", "top": 100, "bottom": 400, "dip": 60, "dipdir": 90,
                "props": {"density": {"value": 3.2, "lower": null, "upper": null}, ...},
                "weight": 10, "fixed": false, "sharp": false},
               {"kind": "layers", "name": "Layers", "shape": "all", "ref": "depth", "top": 0,
                "dip": 0, "dipdir": 90, "origin": [x, y],
                "layers": [{"name": "cover", "thickness": 50, "props": {...}, ...},
                           {"name": "basement", "thickness": null, ...}]},
               {"kind": "boreholes", "name": "Holes", "radius": 0, "weight": 10,
                "holes": [{"name": "BH1", "x": 1000, "y": 900, "collar": null, "azimuth": 0,
                           "dip": 90, "intervals": [{"from": 0, "to": 50, "props": {...}}]}]}],
     "free": {"density": {"value": 2.67, "lower": null, "upper": null},
              "susceptibility": {"lower": null, "upper": null},
              "resistivity": {"value": 100, "lower": null, "upper": null, "mixing": "log"}}}
"""

from __future__ import annotations

import math

import numpy as np

from .geology import DEFAULT_RESISTIVITY, MARGIN, PROPERTIES, build_constraints

DEFAULT_DENSITY = 2.67      # g/cc, the background when the model gives none


def _num(v):
    """A finite number, or None (as the page's mbNum: empty, missing or not a number)."""
    if v is None or (isinstance(v, str) and not v.strip()) or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _plus(v) -> float:
    """JavaScript's unary plus for the numbers the page writes (null and '' are 0)."""
    x = _num(v)
    return 0.0 if x is None else x


def _has(prop: str, v) -> bool:
    return bool(v) and _num(v.get("value")) is not None and (prop != "resistivity" or float(v["value"]) > 0)


def _js(x: float) -> str:
    """A number as JavaScript prints it (the shortest that reads back; 100, not 100.0)."""
    t = repr(float(x))
    return t[:-2] if t.endswith(".0") else t


def _sig(x: float, digits: int = 8) -> float:
    return float(f"{x:.{digits}g}")


def ring(item: dict):
    """An item's outline [[x, y], ...] (m), or None: a stack everywhere."""
    shape = item.get("shape")
    if shape == "box" and item.get("box"):
        w, e, s, n = (float(v) for v in item["box"])
        return [[w, s], [e, s], [e, n], [w, n]]
    if shape == "cylinder" and item.get("cyl"):
        c = item["cyl"]
        x, y, r = float(c["x"]), float(c["y"]), float(c["r"])
        return [[x + r * math.cos(math.pi * k / 16), y + r * math.sin(math.pi * k / 16)] for k in range(32)]
    if shape == "polygon" and item.get("poly") and len(item["poly"]) >= 3:
        return [[float(q[0]), float(q[1])] for q in item["poly"]]
    return None


def _ring_out(r):
    return [[round(x, 2), round(y, 2)] for x, y in r]


def background_density(builder: dict) -> float:
    """The density the contrasts are relative to: the model's free value, else 2.67 g/cc."""
    v = _num(((builder.get("free") or {}).get("density") or {}).get("value"))
    return v if v is not None and v > 0 else DEFAULT_DENSITY


def unit(prop: str, v: dict, owner: dict, background: float) -> dict:
    """The unit of a value ``v`` ({value, lower, upper}) of a body, layer or hole set."""
    m, val = MARGIN[prop], float(v["value"])
    fixed = bool(owner.get("fixed"))
    w = _num(owner.get("weight"))
    w = 5.0 if w is None else w

    def given(x):
        return None if fixed else _num(x)

    if prop == "resistivity":
        f = m ** 0.1 if fixed else m
        lo, hi = given(v.get("lower")), given(v.get("upper"))
        lo = val / f if lo is None else lo
        hi = val * f if hi is None else hi
    else:
        d = m / 10 if fixed else m
        lo, hi = given(v.get("lower")), given(v.get("upper"))
        lo = val - d if lo is None else lo
        hi = val + d if hi is None else hi
    if fixed:
        w = max(w, 100.0)
    lo, hi = min(lo, val), max(hi, val)
    c = (lambda x: x - background) if prop == "density" else (lambda x: x)
    out = {"value": _sig(c(val)), "lower": _sig(c(lo)), "upper": _sig(c(hi)), "weight": w}
    if owner.get("sharp"):
        out["sharp"] = True
    return out


def builder_spec(builder: dict, prop: str) -> dict | None:
    """The geology spec of ``prop`` from a builder, as the upload page submits it; None when no
    part of the model has a value of it."""
    if prop not in PROPERTIES:
        raise ValueError(f"Unknown property '{prop}' (expected {list(PROPERTIES)})")
    bg = background_density(builder)
    units, sources = {}, []
    items = builder.get("items") or []
    for it in items:
        kind = it.get("kind")
        r = ring(it)
        ref = {"reference": "elevation"} if it.get("ref") == "elevation" else {}
        if kind == "body":
            if not r or not _has(prop, (it.get("props") or {}).get(prop)):
                continue
            units[it["name"]] = unit(prop, it["props"][prop], it, bg)
            dip = _plus(it.get("dip", 90))
            sources.append({"type": "body", "unit": it["name"], "crs": "job", "polygon": _ring_out(r),
                            "top_m": _plus(it.get("top")), "bottom_m": _plus(it.get("bottom")), **ref,
                            **({"dip": dip, "dip_direction": _plus(it.get("dipdir"))} if dip < 90 else {})})
        elif kind == "layers":
            any_unit, layers = False, []
            n = len(it.get("layers") or [])
            for j, layer in enumerate(it.get("layers") or []):
                last_open = layer.get("thickness") is None and j == n - 1
                t = None if last_open else _plus(layer.get("thickness"))
                v = (layer.get("props") or {}).get(prop)
                if not _has(prop, v):
                    layers.append({"unit": None, "thickness_m": t})
                    continue
                units[layer["name"]] = unit(prop, v, layer, bg)
                any_unit = True
                layers.append({"unit": layer["name"], "thickness_m": t})
            if not any_unit:
                continue
            dip = _plus(it.get("dip"))
            tilt = {"dip": dip, "dip_direction": _plus(it.get("dipdir")),
                    "origin": [round(_plus(v), 2) for v in it.get("origin") or [0, 0]]} if dip > 0 else {}
            sources.append({"type": "layers", "name": it.get("name"), "crs": "job",
                            "top_m": _plus(it.get("top")), **ref, **tilt,
                            **({"polygon": _ring_out(r)} if r else {}), "layers": layers})
    for bs in items:                  # logged holes: an interval with a value is a unit
        if bs.get("kind") != "boreholes":
            continue
        holes = []
        for h in bs.get("holes") or []:
            intervals = []
            for iv in h.get("intervals") or []:
                v = (iv.get("props") or {}).get(prop)
                a, b = _plus(iv.get("from")), _plus(iv.get("to"))
                if not _has(prop, v) or not b > a:
                    continue
                name = f"{bs['name']}: {h['name']} {_js(a)}–{_js(b)} m"
                units[name] = unit(prop, v, bs, bg)
                intervals.append({"from_m": a, "to_m": b, "unit": name})
            if intervals:
                az, inc = h.get("azimuth"), h.get("dip")
                holes.append({"name": h["name"], "x": _plus(h.get("x")), "y": _plus(h.get("y")),
                              "collar_m": _num(h.get("collar")),
                              "azimuth": _plus(0 if az is None else az),
                              "inclination": _plus(90 if inc is None else inc), "intervals": intervals})
        if holes:
            sources.append({"type": "boreholes", "name": bs.get("name"), "crs": "job",
                            "radius_m": _plus(bs.get("radius")), "holes": holes})
    if not sources:
        return None
    f = (builder.get("free") or {}).get(prop) or {}
    free = {}
    for k in ("lower", "upper"):
        x = _num(f.get(k))
        if x is not None:
            free[k] = round(x - bg, 6) if prop == "density" else x
    if prop == "resistivity" and (_num(f.get("value")) or 0) > 0:
        free["value"] = float(f["value"])
    spec = {"name": f"{len(sources)} drawn part{'s' if len(sources) > 1 else ''}",
            "property": prop, "units": units, "sources": sources, "unconstrained": free}
    if prop == "density":
        spec["background"] = bg
    if prop == "resistivity":
        spec["mixing"] = f.get("mixing") or "log"
    return spec


def model_values(builder: dict, prop: str, centres, half_sizes, surface=None,
                 background_resistivity: float | None = None) -> np.ndarray:
    """``prop`` on cells (their centres and half sizes, (n, 3), m) in model units: density
    contrasts to the model's background density (g/cc), susceptibility (SI), log conductivity
    (ln S/m).  Each cell holds the volume average of the parts in it; the rest of the ground
    is the model's free value (resistivity: ``free.resistivity.value``, else
    ``background_resistivity``, else 100 ohm m; density: the background, 0 contrast;
    susceptibility 0).  ``surface(x, y)`` is the ground's elevation (default: the model's flat
    ground)."""
    centres = np.asarray(centres, dtype=float)
    if surface is None:
        g = _plus(builder.get("ground"))
        surface = lambda x, y: np.full(np.shape(x), g, dtype=float)    # noqa: E731
    rho = _num(((builder.get("free") or {}).get("resistivity") or {}).get("value"))
    rho = rho if rho and rho > 0 else (background_resistivity or DEFAULT_RESISTIVITY)
    free = -math.log(rho) if prop == "resistivity" else 0.0
    spec = builder_spec(builder, prop)
    if spec is None:
        return np.full(len(centres), free)
    geo = build_constraints(spec, centres, half_sizes, surface, default_reference=free)
    return np.asarray(geo.reference, dtype=float)
