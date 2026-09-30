"""Terrain correction (methods/terrain.py) and upward continuation (methods/continuation.py)."""

import numpy as np
import pytest

from geoinv3d.methods.continuation import upward_continue
from geoinv3d.methods.terrain import G_MGAL, line_gz, prism_gz, terrain_correction


def _dem(n=1400, d=90.0):
    x = (np.arange(n) + 0.5) * d
    return x, x.copy(), np.zeros((n, n))


def _hollow_cylinder(a, b, h, density=2.67):
    """Attraction on the axis, at the base level, of a ring of rock a < r < b, h high."""
    return 2 * np.pi * G_MGAL * density * (b - a - np.sqrt(b * b + h * h) + np.sqrt(a * a + h * h))


class TestPrism:
    def test_thin_slab_is_the_bouguer_slab(self):
        # a very wide, thin plate under the station: 2 pi G rho h
        g = prism_gz(-5e6, 5e6, -5e6, 5e6, np.array(10.0)) * G_MGAL * 2.67
        assert g == pytest.approx(0.04193 * 2.67 * 10.0, rel=1e-4)

    def test_rock_above_and_missing_below_count_alike(self):
        up = prism_gz(100.0, 190.0, -40.0, 50.0, np.array(35.0))
        down = prism_gz(100.0, 190.0, -40.0, 50.0, np.array(-35.0))
        assert up > 0 and up == pytest.approx(down)

    def test_station_on_an_edge_or_corner_is_finite(self):
        g = prism_gz(np.array([0.0, -45.0]), np.array([90.0, 45.0]),
                     np.array([0.0, 0.0]), np.array([90.0, 90.0]), np.array([20.0, 20.0]))
        assert np.isfinite(g).all() and (g > 0).all()

    def test_far_prism_matches_the_line_formula(self):
        p = prism_gz(4455.0, 4905.0, 1000.0, 1450.0, np.array(120.0))
        r = np.hypot(4680.0, 1225.0)
        assert line_gz(r, 450.0 ** 2, 120.0 ** 2) == pytest.approx(p, rel=5e-3)


class TestTerrainCorrection:
    def test_flat_ground_needs_none(self):
        x, y, z = _dem()
        z += 512.0
        tc = terrain_correction([63000.0, 63031.0], [63000.0, 62950.0], x, y, z)
        np.testing.assert_allclose(tc, 0.0, atol=1e-9)

    @pytest.mark.parametrize("a, b, h", [(900.0, 2500.0, 150.0),       # inside the near zone
                                         (6000.0, 15000.0, 300.0),     # the middle zone
                                         (30000.0, 50000.0, -400.0)])  # far, and a depression
    def test_ring_plateau_matches_the_hollow_cylinder(self, a, b, h):
        x, y, z = _dem()
        xs, ys = 63020.0, 62990.0                       # not on a cell centre
        r = np.hypot(x[None, :] - xs, y[:, None] - ys)
        z[(r > a) & (r < b)] = h
        tc = terrain_correction([xs], [ys], x, y, z, radius=60000.0)[0]
        assert tc == pytest.approx(_hollow_cylinder(a, b, abs(h)), rel=0.03)

    @pytest.mark.parametrize("xs, ys", [(63000.0, 63000.0), (63031.0, 62977.0)])
    def test_a_slope_under_the_station_has_no_staircase_slab(self, xs, ys):
        # A plane through the station with slope t: within a rectangle around the station
        # the correction is G rho t^2 / 2 x the integral of x^2 / r^3 (thin-layer limit);
        # flat prisms next to the station would add a slab of about 2 pi G rho x 1.5 m
        x, y, z = _dem()
        t = 0.1
        z += t * (x[None, :] - x.mean())
        parts = terrain_correction([xs], [ys], x, y, z, return_parts=True)[1]
        i, j = int(ys // 90), int(xs // 90)
        w, e = (j - 5) * 90.0 - xs, (j + 6) * 90.0 - xs
        s, n = (i - 5) * 90.0 - ys, (i + 6) * 90.0 - ys
        u = (np.arange(4000) + 0.5) / 4000
        xx, yy = (w + (e - w) * u)[None, :], (s + (n - s) * u)[:, None]
        integral = (xx ** 2 / (xx ** 2 + yy ** 2) ** 1.5).mean() * (e - w) * (n - s)
        assert parts[0, 0] == pytest.approx(G_MGAL * 2.67 * t * t / 2 * integral, rel=0.03)

    def test_stations_near_the_edge_are_not_computed(self):
        x, y, z = _dem()
        tc = terrain_correction([5000.0, 63000.0], [63000.0, 63000.0], x, y, z)
        assert np.isnan(tc[0]) and tc[1] == 0.0

    def test_gaps_are_refused(self):
        x, y, z = _dem(200)
        z[3, 3] = np.nan
        with pytest.raises(ValueError, match="gaps"):
            terrain_correction([9000.0], [9000.0], x, y, z, radius=1000.0)


class TestUpwardContinuation:
    def test_point_source_field_at_a_new_height(self):
        # the vertical field of a point mass, 1 km below the plane, seen 500 m higher
        d = 100.0
        c = (np.arange(400) - 199.5) * d
        xx, yy = np.meshgrid(c, c)

        def field(depth):
            return depth / (xx ** 2 + yy ** 2 + depth ** 2) ** 1.5

        up = upward_continue(field(1000.0), d, d, 500.0)
        inner = slice(100, 300)
        np.testing.assert_allclose(up[inner, inner], field(1500.0)[inner, inner],
                                   atol=0.01 * field(1500.0).max())

    def test_a_constant_and_zero_height_are_unchanged(self):
        g = np.full((50, 60), 7.0)
        np.testing.assert_allclose(upward_continue(g, 10.0, 10.0, 80.0), 7.0)
        r = np.random.default_rng(0).normal(size=(40, 40))
        np.testing.assert_array_equal(upward_continue(r, 10.0, 10.0, 0.0), r)

    def test_short_wavelengths_are_damped_by_exp_minus_kh(self):
        d, n = 50.0, 256
        x = np.arange(n) * d
        lam = n * d / 16                                  # periodic in the grid
        g = np.sin(2 * np.pi * x / lam)[None, :] * np.ones((n, 1))
        up = upward_continue(g, d, d, 200.0, pad=n)
        ratio = np.abs(up[n // 2, n // 4:3 * n // 4]).max()      # away from the faded edges
        assert ratio == pytest.approx(np.exp(-2 * np.pi * 200.0 / lam), rel=0.03)

    def test_downward_is_refused(self):
        with pytest.raises(ValueError, match="height"):
            upward_continue(np.zeros((8, 8)), 1.0, 1.0, -5.0)
