"""Models of the model builder in Python (geoinv3d.methods.builder): the spec the upload page
submits for them, and their values on a mesh."""

import math

import numpy as np
import pytest

from geoinv3d.methods.builder import builder_spec, model_values, ring


def _p(density=None, susceptibility=None, resistivity=None, **bounds):
    out = {k: {"value": v, "lower": None, "upper": None}
           for k, v in (("density", density), ("susceptibility", susceptibility), ("resistivity", resistivity))}
    for k, v in bounds.items():
        prop, side = k.rsplit("_", 1)
        out[prop][side] = v
    return out


def _body(**kw):
    return {"kind": "body", "name": "ore", "shape": "box", "box": [0, 100, 0, 100], "ref": "depth",
            "top": 0, "bottom": 50, "dip": 90, "dipdir": 90, "props": _p(3.0, 0.05, 10.0),
            "weight": 10, "fixed": False, "sharp": False, **kw}


def _model(*items, density=2.5, resistivity=None):
    return {"version": 2, "ground": 0, "items": list(items),
            "free": {"density": {"value": density, "lower": None, "upper": None},
                     "susceptibility": {"lower": None, "upper": None},
                     "resistivity": {"value": resistivity, "lower": None, "upper": None, "mixing": "log"}}}


def test_a_body_is_a_unit_with_the_pages_margins():
    spec = builder_spec(_model(_body(sharp=True)), "density")
    assert spec["property"] == "density" and spec["background"] == 2.5
    assert spec["units"] == {"ore": {"value": 0.5, "lower": 0.45, "upper": 0.55, "weight": 10.0, "sharp": True}}
    src = spec["sources"][0]
    assert src == {"type": "body", "unit": "ore", "crs": "job", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                   "top_m": 0.0, "bottom_m": 50.0}          # vertical: no dip
    dipping = builder_spec(_model(_body(dip=60, dipdir=45, ref="elevation", top=-10, bottom=-200)), "density")
    s = dipping["sources"][0]
    assert (s["dip"], s["dip_direction"], s["reference"]) == (60.0, 45.0, "elevation")
    # resistivity: x / and / 2; susceptibility +- 0.005
    r = builder_spec(_model(_body()), "resistivity")
    assert r["units"]["ore"] == {"value": 10.0, "lower": 5.0, "upper": 20.0, "weight": 10.0}
    assert r["mixing"] == "log" and "background" not in r
    assert builder_spec(_model(_body()), "susceptibility")["units"]["ore"]["lower"] == pytest.approx(0.045)


def test_fixed_and_given_ranges():
    b = _body(fixed=True, weight=3, props=_p(3.0, None, 10.0, density_lower=2.0, resistivity_upper=1000))
    d = builder_spec(_model(b), "density")["units"]["ore"]
    assert d["weight"] == 100.0                               # fixed: at least 100
    assert (d["lower"], d["upper"]) == (pytest.approx(0.495), pytest.approx(0.505))   # given range ignored
    r = builder_spec(_model(b), "resistivity")["units"]["ore"]
    assert r["upper"] == pytest.approx(10 * 2 ** 0.1, rel=1e-7)
    given = builder_spec(_model(_body(props=_p(3.0, density_lower=2.9, density_upper=3.2))), "density")
    assert given["units"]["ore"]["lower"] == pytest.approx(0.4) and given["units"]["ore"]["upper"] == pytest.approx(0.7)
    # a missing weight is the default 5 (as the page)
    assert builder_spec(_model(_body(weight=None)), "density")["units"]["ore"]["weight"] == 5.0
    # nothing with the property: no spec
    assert builder_spec(_model(_body(props=_p(resistivity=10))), "density") is None
    with pytest.raises(ValueError):
        builder_spec(_model(), "porosity")


def test_layer_stacks():
    stack = {"kind": "layers", "name": "Stack", "shape": "box", "box": [0, 1000, 0, 500], "ref": "depth",
             "top": 0, "dip": 5, "dipdir": 90, "origin": [500, 250],
             "layers": [{"name": "cover", "thickness": 40, "props": _p(resistivity=30), "weight": 5},
                        {"name": "clay", "thickness": 15, "props": _p(), "weight": 5},
                        {"name": "base", "thickness": None, "props": _p(resistivity=3000), "weight": 5}]}
    spec = builder_spec(_model(stack), "resistivity")
    src = spec["sources"][0]
    assert src["layers"] == [{"unit": "cover", "thickness_m": 40.0}, {"unit": None, "thickness_m": 15.0},
                             {"unit": "base", "thickness_m": None}]
    assert (src["dip"], src["dip_direction"], src["origin"], src["name"]) == (5.0, 90.0, [500.0, 250.0], "Stack")
    assert src["polygon"] == [[0, 0], [1000, 0], [1000, 500], [0, 500]]
    assert set(spec["units"]) == {"cover", "base"}
    assert builder_spec(_model(stack), "density") is None     # no layer has a density
    flat = dict(stack, shape="all", dip=0)
    s = builder_spec(_model(flat), "resistivity")["sources"][0]
    assert "polygon" not in s and "dip" not in s


def test_logged_holes():
    holes = {"kind": "boreholes", "name": "Holes", "radius": 50, "weight": 20, "fixed": False, "sharp": False,
             "holes": [{"name": "BH1", "x": 10, "y": 20, "collar": None, "azimuth": 30, "dip": 70,
                        "intervals": [{"from": 0, "to": 12.5, "props": _p(resistivity=8)},
                                      {"from": 12.5, "to": 40, "props": _p(density=2.9)},
                                      {"from": 40, "to": 40, "props": _p(resistivity=8)}]},
                       {"name": "BH2", "x": 0, "y": 0, "intervals": [{"from": 0, "to": 5, "props": _p()}]}]}
    spec = builder_spec(_model(holes), "resistivity")
    assert spec["units"] == {"Holes: BH1 0–12.5 m": {"value": 8.0, "lower": 4.0, "upper": 16.0, "weight": 20.0}}
    src = spec["sources"][0]
    assert src["radius_m"] == 50.0 and len(src["holes"]) == 1          # BH2 has no value
    # numbers in the names as the page prints them: 0, 12.5, 101.4984 (not 101.498)
    deep = dict(holes, holes=[dict(holes["holes"][0], intervals=[{"from": 101.4984, "to": 152.4384,
                                                                 "props": _p(resistivity=8)}])])
    assert list(builder_spec(_model(deep), "resistivity")["units"]) == ["Holes: BH1 101.4984–152.4384 m"]
    h = src["holes"][0]
    assert (h["azimuth"], h["inclination"], h["collar_m"]) == (30.0, 70.0, None)
    assert h["intervals"] == [{"from_m": 0.0, "to_m": 12.5, "unit": "Holes: BH1 0–12.5 m"}]


def test_values_on_cells():
    # 50 m cells over 200 m x 200 m x 100 m; the body fills one column of 2 x 2 x 1 cells
    x = np.arange(25.0, 200, 50)
    z = np.array([-25.0, -75.0])
    cx, cy, cz = (a.ravel() for a in np.meshgrid(x, x, z, indexing="ij"))
    c, h = np.column_stack([cx, cy, cz]), np.full((cx.size, 3), 25.0)
    v = model_values(_model(_body()), "density", c, h)
    inside = (cx < 100) & (cy < 100) & (cz > -50)
    assert np.allclose(v[inside], 0.5) and np.allclose(v[~inside], 0.0)
    # half a cell deep: half the contrast; resistivity mixes in log: 10 and 1000 -> 100
    half = model_values(_model(_body(bottom=25)), "density", c, h)
    assert np.allclose(half[inside], 0.25)
    r = model_values(_model(_body(bottom=25), resistivity=1000.0), "resistivity", c, h)
    assert np.allclose(-r[inside], math.log(100.0)) and np.allclose(-r[~inside], math.log(1000.0))
    # nothing with the property: the free value everywhere
    assert np.allclose(model_values(_model(), "resistivity", c, h, background_resistivity=50.0), -math.log(50.0))


def test_ring_shapes():
    assert len(ring({"shape": "cylinder", "cyl": {"x": 0, "y": 0, "r": 10}})) == 32
    assert ring({"shape": "polygon", "poly": [[0, 0], [1, 0]]}) is None
    assert ring({"shape": "all"}) is None
