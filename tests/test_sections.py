"""Slices of a result at its mesh's resolution (viz/sections.py)."""

import numpy as np
import pytest

from geoinv3d.cloud.worker import pack_result
from geoinv3d.viz.result_workflow import load_result
from geoinv3d.viz.sections import ResultModel
from tests.test_result_workflow import _run


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    _, zpath, _ = _run(tmp_path_factory.mktemp("sec"))
    return ResultModel(load_result(zpath))


def _arr(values):
    return np.array([[np.nan if v is None else v for v in row] for row in values], float)


def test_a_line_samples_the_cells_it_crosses(model):
    W, E, S, N = model.extent
    y = (S + N) / 2
    d = model.line(W, y, E, y)
    v = _arr(d["values"])
    assert v.shape == (len(d["v"]), len(d["h"]))
    assert d["h"][1] - d["h"][0] == pytest.approx(model.h / 2)          # half a cell
    # the same values as the mesh at those points
    i, k = len(d["h"]) // 2, len(d["v"]) // 3
    x = W + d["h"][i]
    z = float(model.ground(np.array([x]), np.array([y]))[0]) - d["v"][k]
    assert v[k, i] == pytest.approx(model.sample(np.array([[x, y, z]]))[0], rel=1e-3)


def test_elevation_sections_leave_the_air_blank(model):
    W, E, S, N = model.extent
    d = model.line(W, S, E, N, ref="elev")
    assert d["v"][0] > d["v"][-1] and len(d["ground"]) == len(d["h"])


def test_a_plan_slice_and_the_strike(model):
    d = model.plan(150.0)
    assert _arr(d["values"]).shape == (len(d["y"]), len(d["x"]))
    p = model.strike_profile()
    W, E, S, N = model.extent
    for x, y in (p["a"], p["b"]):
        assert W - 1e-6 <= x <= E + 1e-6 and S - 1e-6 <= y <= N + 1e-6
    assert 0 <= p["strike_deg"] < 180


def test_the_strike_of_an_elongated_body():
    """A body striking 45° east of north: the profile runs across it (NW to SE)."""
    import math

    class Fake(ResultModel):
        def __init__(self):
            self.extent, self.depth_core, self.h = [0.0, 1000.0, 0.0, 1000.0], 600.0, 10.0

        def plan(self, level, ref="ground", extent=None):
            xs = ys = list(np.arange(5.0, 1000.0, 10.0))
            X, Y = np.meshgrid(xs, ys)
            v = np.exp(-((X - Y) ** 2) / (2 * 40.0 ** 2))       # a ridge along x = y
            return {"x": xs, "y": ys, "values": v.tolist()}

    p = Fake().strike_profile()
    assert p["strike_deg"] == pytest.approx(45.0, abs=2.0)
    (ax, ay), (bx, by) = p["a"], p["b"]
    assert math.degrees(math.atan2(by - ay, bx - ax)) == pytest.approx(-45.0, abs=2.0)   # across it


def test_the_3d_view_and_the_slices_share_one_colour_scale(tmp_path):
    """The 3D view's range and sign are the slices' (the 1st-99th percentile, both signs or
    not), and its depth is theirs, so a cell looks the same in both."""
    from geoinv3d.viz.result_workflow import build_workflow, model_signed, robust_range
    _, zpath, _ = _run(tmp_path)
    meta = load_result(zpath)
    rm = ResultModel(meta)
    node = next(n for n in build_workflow([meta])["nodes"] if n["type"] == "RegularizedInversionNode")
    m3 = node["output"]["model_3d"]
    assert m3["range"] == pytest.approx(rm.range) and m3["signed"] == rm.signed
    assert m3["depth_max"] == pytest.approx(rm.depth_core)
    # a few extreme cells do not set the scale; both signs only when both are there
    v = np.r_[np.linspace(0.0, 1.0, 1000), 500.0]
    assert robust_range(v)[1] < 1.0 and not model_signed(v) and model_signed(np.r_[v, -0.5])
    assert not model_signed(-v[1:])          # all one sign (e.g. a log conductivity)
