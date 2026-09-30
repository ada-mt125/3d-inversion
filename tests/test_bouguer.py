"""Telling a simple from a terrain-corrected Bouguer anomaly (methods/bouguer.py)."""

import numpy as np
import pytest

from geoinv3d.methods.bouguer import (
    FREE_AIR, SLAB, check_table, check_terrain_correction, normal_gravity_grs80, to_mgal,
)


def _stations(seed=0):
    """40 x 40 stations 500 m apart over plains with two ranges of hills."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(40) * 500.0, np.arange(40) * 500.0)
    x, y = xx.ravel() + rng.uniform(-100, 100, 1600), yy.ravel() + rng.uniform(-100, 100, 1600)
    hills = sum(a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w ** 2))
                for a, cx, cy, w in ((450, 5000, 6000, 1500), (350, 14000, 13000, 1200)))
    h = 400 + 0.01 * x + hills + rng.normal(0, 5, 1600)
    lat = 15.0 + y / 111e3
    gamma = normal_gravity_grs80(lat)
    anomaly = 20 * np.sin(x / 7000) + 10 * np.cos(y / 5000)      # the geology
    g_obs = gamma - FREE_AIR * h + SLAB * 2.67 * h + anomaly + rng.normal(0, 0.05, 1600)
    return x, y, h, lat, gamma, g_obs


def _relief(x, y, h, radius=3000.0):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    return np.array([np.ptp(h[nb]) for nb in tree.query_ball_point(np.column_stack([x, y]), radius)])


def _anomaly(x, y, h, gamma, g_obs, rho=2.67, tc=None, rounding=None):
    ba = g_obs - gamma + FREE_AIR * h - SLAB * rho * h + (0 if tc is None else tc)
    return np.round(ba / rounding) * rounding if rounding else ba


class TestCheck:
    def test_simple_bouguer_anomaly(self):
        x, y, h, lat, gamma, g_obs = _stations()
        r = check_terrain_correction(x, y, h, _anomaly(x, y, h, gamma, g_obs), g_obs, gamma)
        assert r["verdict"] == "none" and r["reduction_density"] == pytest.approx(2.67, abs=0.005)
        assert "No terrain correction" in r["message"]

    def test_terrain_correction_is_detected(self):
        # a terrain correction of up to ~4 mGal, larger where the ground is rough
        x, y, h, lat, gamma, g_obs = _stations()
        tc = 0.01 * _relief(x, y, h)
        r = check_terrain_correction(x, y, h, _anomaly(x, y, h, gamma, g_obs, tc=tc), g_obs, gamma)
        assert r["verdict"] == "applied", r
        assert r["rough_minus_smooth_mgal"] > 1.0 and r["relief_corr"] > 0.15
        # the density comes from the flatter half, where the correction is small
        assert r["reduction_density"] == pytest.approx(2.67, abs=0.03)

    def test_rounded_to_one_mgal_like_ngpm(self):
        x, y, h, lat, gamma, g_obs = _stations()
        g_r, gamma_r = np.round(g_obs), np.round(gamma)
        simple = _anomaly(x, y, h, gamma, g_obs, rounding=1.0)
        assert check_terrain_correction(x, y, h, simple, g_r, gamma_r)["verdict"] == "none"
        tc = 0.01 * _relief(x, y, h)
        complete = _anomaly(x, y, h, gamma, g_obs, tc=tc, rounding=1.0)
        assert check_terrain_correction(x, y, h, complete, g_r, gamma_r)["verdict"] == "applied"

    def test_other_reduction_density(self):
        x, y, h, lat, gamma, g_obs = _stations()
        r = check_terrain_correction(x, y, h, _anomaly(x, y, h, gamma, g_obs, rho=2.6), g_obs, gamma)
        assert r["reduction_density"] == pytest.approx(2.6, abs=0.005) and r["verdict"] == "none"

    def test_normal_gravity_from_latitude_and_units(self):
        x, y, h, lat, gamma, g_obs = _stations()
        ba = _anomaly(x, y, h, gamma, g_obs)
        # observed gravity in m/s^2, normal gravity from the latitude (GRS80)
        r = check_terrain_correction(x, y, h, ba, g_obs * 1e-5, lat=lat)
        assert r["verdict"] == "none" and r["reduction_density"] == pytest.approx(2.67, abs=0.005)
        assert to_mgal([9.78])[0] == pytest.approx(978000) and to_mgal([978.0])[0] == 978000

    def test_too_few_stations(self):
        x, y, h, lat, gamma, g_obs = _stations()
        r = check_terrain_correction(x[:20], y[:20], h[:20], g_obs[:20] * 0, g_obs[:20], gamma[:20])
        assert r["verdict"] == "unclear"


class TestTable:
    def _write(self, path, cols, rows):
        path.write_text(",".join(cols) + "\n" + "\n".join(",".join(f"{v:.6f}" for v in r) for r in rows) + "\n")

    def test_ngpm_style_table(self, tmp_path):
        x, y, h, lat, gamma, g_obs = _stations()
        lon = 76.0 + x / 107e3
        ba = _anomaly(x, y, h, gamma, g_obs)
        p = tmp_path / "stations.csv"
        self._write(p, ["X", "Y", "bouguer_an", "elevation", "observed_g", "theoretical_g"],
                    np.column_stack([lon, lat, ba, h, g_obs, gamma]))
        r = check_table(p)
        assert r["verdict"] == "none" and r["reduction_density"] == pytest.approx(2.67, abs=0.01)

    def test_latitude_instead_of_normal_gravity(self, tmp_path):
        x, y, h, lat, gamma, g_obs = _stations()
        ba = _anomaly(x, y, h, gamma, g_obs, tc=0.01 * _relief(x, y, h))
        p = tmp_path / "s.csv"
        self._write(p, ["lon", "lat", "ba", "elev", "gobs"],
                    np.column_stack([76.0 + x / 107e3, lat, ba, h, g_obs]))
        assert check_table(p)["verdict"] == "applied"

    def test_table_without_observed_gravity(self, tmp_path):
        p = tmp_path / "s.csv"
        p.write_text("x,y,gz\n1,2,3\n4,5,6\n")
        assert check_table(p) is None
