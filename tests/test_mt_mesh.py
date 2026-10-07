"""MT meshes from the skin depths (geoinv3d.cloud.meshing, worker._mt_axes) and where the
fields are measured (MTMethod.measurement_points)."""

import numpy as np
import pytest
from discretize import TensorMesh

from geoinv3d.cloud.meshing import (
    MT_PAD_SKIN_DEPTHS, MT_TOP_LAYERS, MT_TOP_PER_SKIN_DEPTH, MT_Z_GROWTH, mt_rho_stats,
    recommend_mt_mesh, skin_depth,
)
from geoinv3d.methods.mt import MTMethod



def _flat(x, y):
    return np.zeros_like(np.asarray(x, dtype=float))


def _halfspace_stats(freqs, rho):
    return {"frequencies": list(freqs), "p10": [rho] * len(freqs), "p50": [rho] * len(freqs),
            "p90": [rho] * len(freqs)}


def test_skin_depth():
    assert skin_depth(100.0, 1.0) == pytest.approx(5032.9, rel=1e-4)    # 503 sqrt(rho / f)


def test_rho_stats_take_percentiles_per_frequency():
    rho = np.array([[1.0, 2.0, 3.0, np.nan], [np.nan, np.nan, np.nan, np.nan], [10, 20, 30, 40]])
    st = mt_rho_stats([10.0, 1.0, 0.1], rho)
    assert st["frequencies"] == [0.1, 10.0]          # sorted, the empty frequency left out
    assert st["p50"] == pytest.approx([25.0, 2.0])


def test_halfspace_design_follows_the_skin_depths():
    freqs = np.logspace(-1, 3, 9)
    r = recommend_mt_mesh((0, 10000, 0, 10000), 2000, freqs, _halfspace_stats(freqs, 100.0))
    d_min, d_max = skin_depth(100.0, 1000.0), skin_depth(100.0, 0.1)
    assert d_min / (2.5 * MT_TOP_PER_SKIN_DEPTH) <= r["core_cell_z_m"] <= d_min / MT_TOP_PER_SKIN_DEPTH
    assert r["pad_distance_m"] >= MT_PAD_SKIN_DEPTHS * d_max and r["air_m"] == r["pad_distance_m"]
    assert r["depth_core_m"] >= d_max / np.sqrt(2)                       # the Bostick depth
    assert r["core_cell_m"] == 1000.0                                    # half the spacing
    assert r["z_growth"] == MT_Z_GROWTH and r["top_layers"] == MT_TOP_LAYERS


def test_deep_resistivity_rises_with_a_climbing_curve():
    """Niblett-Bostick: rho_a still climbing at the lowest frequency means more resistive rock
    below, so the padding grows; a falling curve keeps rho_a."""
    freqs = [0.3, 3.0, 30.0]
    climbing = {"frequencies": freqs, "p10": [196, 37, 8], "p50": [196, 37, 8], "p90": [196, 37, 8]}
    falling = {"frequencies": freqs, "p10": [19, 54, 296], "p50": [19, 54, 296], "p90": [19, 54, 296]}
    up = recommend_mt_mesh((0, 2400, 0, 2400), 1200, freqs, climbing)
    down = recommend_mt_mesh((0, 2400, 0, 2400), 1200, freqs, falling)
    assert 600 < up["rho_deep"] < 2000           # the basement of 10 over 1000 ohm m
    assert down["rho_deep"] == pytest.approx(19.0)
    assert up["rho_low"] == pytest.approx(8.0)   # the top cell from the least resistive


def test_no_impedance_uses_the_background():
    r = recommend_mt_mesh((0, 2000, 0, 2000), 500, [1.0, 10.0], None, rho_background=50.0)
    assert r["rho_low"] == r["rho_deep"] == 50.0


@pytest.mark.parametrize("octree", [False, True])
def test_mt_axes(octree):
    from geoinv3d.cloud.worker import _mt_axes
    freqs = [0.1, 1.0, 10.0, 100.0]
    r = recommend_mt_mesh((0, 6000, 0, 4000), 2000, freqs, _halfspace_stats(freqs, 100.0))
    hx, hy, hz, origin = _mt_axes((0, 6000, 0, 4000), _flat, False, r, octree=octree)
    nodes = origin[2] + np.r_[0.0, np.cumsum(hz)]
    g = int(np.argmin(np.abs(nodes)))
    assert abs(nodes[g]) < 1e-6                                        # the ground is a node
    top = r["core_cell_z_m"]
    np.testing.assert_allclose(hz[g - MT_TOP_LAYERS:g + 2], top)       # top cells either side
    below = hz[:g][::-1]
    core = np.cumsum(below) <= r["depth_core_m"]
    assert np.all(below[1:][core[1:]] / below[:-1][core[1:]] <= MT_Z_GROWTH + 1e-9)
    assert -origin[2] >= r["pad_distance_m"] and nodes[-1] >= r["air_m"]
    for h, o, lo, hi in ((hx, origin[0], 0, 6000), (hy, origin[1], 0, 4000)):
        assert lo - o >= r["pad_distance_m"] and o + h.sum() - hi >= r["pad_distance_m"]
    if octree:
        assert all(n & (n - 1) == 0 for n in (hx.size, hy.size, hz.size))   # powers of two
        assert g == hz.size // 2                  # the ground stays a face at every level


def test_octree_is_finest_over_the_stations():
    from geoinv3d.cloud.worker import _build_mt_octree_mesh
    freqs = [0.3, 3.0, 30.0]
    r = recommend_mt_mesh((0, 2400, 0, 2400), 1200, freqs, _halfspace_stats(freqs, 100.0))
    tree = _build_mt_octree_mesh((0, 2400, 0, 2400), _flat, False, r).to_discretize()
    cc, h = tree.cell_centers, tree.h_gridded
    under = (np.abs(cc[:, 0] - 1200) < 1200) & (np.abs(cc[:, 1] - 1200) < 1200) & (cc[:, 2] < 0) & (cc[:, 2] > -500)
    assert np.allclose(h[under, 0], r["core_cell_m"])
    far = np.abs(cc[:, 0] - 1200) > 10 * r["core_cell_m"]
    assert h[far, 0].min() > r["core_cell_m"]                          # coarsened away
    # no cell straddles the ground
    zb = cc[:, 2] - h[:, 2] / 2
    assert not np.any((zb < -1e-6) & (zb + h[:, 2] > 1e-6))


def test_measurement_points_on_the_ground_and_in_the_air():
    """E on the top of the ground cell under the station (the staircase of a DEM), H at the
    centre of the air cell above it."""
    tm = TensorMesh([[(100.0, 4)], [(100.0, 4)], [(50.0, 6), (20.0, 2), (80.0, 2)]], origin=[0.0, 0.0, -300.0])
    cc = tm.cell_centers
    ground = np.where(cc[:, 0] < 200, 0.0, -50.0)       # a step: 0 m in the west, -50 m east
    active = cc[:, 2] < ground
    st = np.array([[50.0, 50.0, 10.0], [350.0, 50.0, -60.0]])
    e, h = MTMethod.measurement_points(tm, st, active)
    np.testing.assert_allclose(e[:, 2], [0.0, -50.0])
    np.testing.assert_allclose(h[:, 2], [10.0, -25.0])  # 20 m air cell over 0; 50 m over -50
    np.testing.assert_allclose(e[:, :2], st[:, :2])


@pytest.mark.parametrize("rho, thick", [([10.0, 1000.0], [300.0]), ([1000.0, 10.0], [1000.0]),
                                        ([100.0, 10.0, 1000.0], [500.0, 2000.0])])
def test_smooth_1d_layers_fit_the_curve(rho, thick):
    """The primary's layering: the smoothest that fits the median curve to 2 %."""
    from geoinv3d.cloud.meshing import _rho_a_1d, smooth_1d_layers
    f = np.logspace(-2, 2, 9)
    ra = _rho_a_1d(f, rho, thick)
    tops, sig = smooth_1d_layers({"frequencies": list(f), "p50": list(ra)})
    assert tops[0] == 0 and np.all(np.diff(tops) > 0) and np.all(sig > 0)
    fit = _rho_a_1d(f, 1 / sig, np.diff(tops))
    assert np.sqrt(np.mean(np.log(fit / ra) ** 2)) < 0.025


def test_layered_primary_on_the_mesh():
    """primary_layers: each ground layer of the mesh takes the layer its centre lies in, the
    air stays air."""
    tm = TensorMesh([[(100.0, 2)], [(100.0, 2)], [(100.0, 6), (100.0, 2)]], origin=[0.0, 0.0, -600.0])
    active = tm.cell_centers[:, 2] < 0
    mt = MTMethod(sigma_background=1e-2, primary_layers=([0.0, 250.0], [0.1, 0.001]))
    np.testing.assert_allclose(mt.primary_1d(tm, active), [1e-3] * 4 + [0.1] * 2 + [1e-8] * 2)
    with pytest.raises(ValueError, match="primary_layers"):
        MTMethod(primary_layers=([10.0, 250.0], [0.1, 0.001]))
