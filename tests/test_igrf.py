"""IGRF-14: the coefficients, the synthesis, and the declination from grid north."""

import math

import numpy as np
import pytest

from geoinv3d.methods import igrf as I


def test_coefficients():
    epochs, g, h, sg, sh = I.coefficients()
    assert epochs[0] == 1900 and epochs[-1] == 2025 and g.shape[1] == 14
    assert g[list(epochs).index(2020.0), 1, 0] == -29403.41
    assert sg[1, 0] == 12.6
    g27, _ = I.gauss_coefficients(2027.0)
    assert g27[1, 0] == pytest.approx(-29350.0 + 2 * 12.6)
    g22, _ = I.gauss_coefficients(2022.5)          # halfway between 2020 and 2025
    assert g22[1, 0] == pytest.approx((-29403.41 - 29350.0) / 2)
    with pytest.raises(ValueError):
        I.gauss_coefficients(2031.0)


def test_dates():
    assert I.decimal_year("2020-01-01") == 2020.0
    assert I.decimal_year("2021-07-02") == pytest.approx(2021.5, abs=0.003)
    assert I.decimal_year(2024.25) == 2024.25


def _potential_field(lon, lat_c, r, year):
    """X, Y, Z (geocentric north, east, down) from finite differences of the scalar potential,
    with scipy's associated Legendre functions: independent of the recursion in igrf()."""
    from scipy.special import lpmv
    g, h = I.gauss_coefficients(year)
    a = I.EARTH_RADIUS_KM

    def V(r, theta, phi):
        v = 0.0
        for n in range(1, g.shape[1]):
            for m in range(n + 1):
                schmidt = 1.0 if m == 0 else math.sqrt(2 * math.factorial(n - m) / math.factorial(n + m))
                P = schmidt * (-1) ** m * lpmv(m, n, math.cos(theta))      # no Condon-Shortley phase
                v += a * (a / r) ** (n + 1) * (g[n, m] * math.cos(m * phi) + h[n, m] * math.sin(m * phi)) * P
        return v

    th, ph, e = math.radians(90 - lat_c), math.radians(lon), 1e-6
    Br = -(V(r + 1e-3, th, ph) - V(r - 1e-3, th, ph)) / 2e-3
    Bt = -(V(r, th + e, ph) - V(r, th - e, ph)) / (2 * e) / r
    Bp = -(V(r, th, ph + e) - V(r, th, ph - e)) / (2 * e) / (r * math.sin(th))
    return -Bt, Bp, -Br


@pytest.mark.parametrize("lon, lat", [(76.6, 15.1), (-105.3, 40.0), (151.2, -33.9), (10.0, 70.0)])
def test_the_synthesis_matches_the_potential(lon, lat):
    pytest.importorskip("scipy")
    # on the sphere (geocentric latitude, r = a + h is not the ellipsoid): compare the field
    # igrf() gives before turning it to geodetic axes, via a point whose geodetic and
    # geocentric frames differ by the small angle the function itself computes
    f = I.igrf(lon, lat, 0.0, 2020.0)
    colat = math.radians(90 - lat)
    ct, st = math.cos(colat), math.sin(colat)
    one, two = I.WGS84_A2 * st * st, I.WGS84_B2 * ct * ct
    rho = math.sqrt(one + two)
    r = math.sqrt((I.WGS84_A2 * one + I.WGS84_B2 * two) / (one + two))
    cd, sd = rho / r, (I.WGS84_A2 - I.WGS84_B2) / rho * ct * st / r
    lat_c = 90 - math.degrees(math.atan2(st * cd + ct * sd, ct * cd - st * sd))
    X, Y, Z = _potential_field(lon, lat_c, r, 2020.0)
    Xg, Zg = X * cd + Z * sd, Z * cd - X * sd
    assert f["X"] == pytest.approx(Xg, abs=0.5) and f["Y"] == pytest.approx(Y, abs=0.5)
    assert f["Z"] == pytest.approx(Zg, abs=0.5)


def test_karnataka_from_grid_north():
    """The inducing field of the Karnataka magnetic study (IGRF 2020 at its centre, UTM 43N):
    42,100 nT, I 19.3°, D -1.4° from grid north."""
    pytest.importorskip("rasterio")
    f = I.inducing_field(676000, 1669000, "EPSG:32643", alt_m=500, date=2020.0)
    assert f["F"] == pytest.approx(42094, abs=5) and f["I"] == pytest.approx(19.27, abs=0.02)
    assert f["D"] == pytest.approx(-0.93, abs=0.02)
    assert f["convergence"] == pytest.approx(0.43, abs=0.01)     # true north is west of grid north
    assert f["inducing_field"] == [42094.0, 19.27, -1.35]
    lonlat = I.inducing_field(f["lon"], f["lat"], "EPSG:4326", alt_m=500, date=2020.0)
    assert lonlat["D_grid"] == pytest.approx(f["D"], abs=1e-9) and lonlat["convergence"] == 0.0


def test_field_strength_falls_with_height():
    low, high = I.igrf(76.6, 15.1, 0.0, 2025.0), I.igrf(76.6, 15.1, 10.0, 2025.0)
    assert high["F"] < low["F"] and low["F"] - high["F"] == pytest.approx(3 * low["F"] * 10 / 6371, rel=0.2)
