"""The International Geomagnetic Reference Field (IGRF-14) at a place and date.

The inducing field of a magnetic inversion: its strength F (nT), inclination I and
declination D (degrees).  The coefficients are IAGA's (``geoinv3d/data/igrf14coeffs.txt``:
1900-2025 every five years, interpolated linearly between epochs, and the secular
variation for 2025-2030).  The synthesis follows the IAGA reference program
(igrf14syn): geodetic position on the WGS 84 ellipsoid, field on the geocentric sphere
of radius 6371.2 km, components turned back to geodetic north and down.

For data in a projected CRS the declination the inversion needs is measured from grid
north (the CRS's y axis), not true north: ``inducing_field`` returns both, with the
meridian convergence between them.
"""

from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

COEFFS = Path(__file__).resolve().parents[1] / "data" / "igrf14coeffs.txt"
EARTH_RADIUS_KM = 6371.2
WGS84_A2, WGS84_B2 = 40680631.6, 40408296.0     # semi-axes squared, km^2 (igrf14syn)


@lru_cache(maxsize=1)
def coefficients():
    """(epochs, g, h, sv_g, sv_h): g[k, n, m] for epoch k, nT; sv for after the last epoch."""
    epochs, rows = None, []
    for line in COEFFS.read_text().splitlines():
        parts = line.split()
        if not parts or line.startswith("#"):
            continue
        if parts[0] == "g/h":
            epochs = [float(p) for p in parts[3:-1]]
            continue
        if parts[0] in ("g", "h"):
            rows.append(parts)
    nmax = max(int(r[1]) for r in rows)
    k = len(epochs)
    g = np.zeros((k, nmax + 1, nmax + 1))
    h = np.zeros_like(g)
    sg, sh = np.zeros((nmax + 1, nmax + 1)), np.zeros((nmax + 1, nmax + 1))
    for r in rows:
        n, m = int(r[1]), int(r[2])
        vals = [float(v) for v in r[3:3 + k]]
        sv = float(r[3 + k]) if len(r) > 3 + k else 0.0
        if r[0] == "g":
            g[:, n, m], sg[n, m] = vals, sv
        else:
            h[:, n, m], sh[n, m] = vals, sv
    return np.array(epochs), g, h, sg, sh


def decimal_year(date) -> float:
    """A date (datetime.date, "YYYY-MM-DD" or a decimal year) as a decimal year."""
    if isinstance(date, (int, float)):
        return float(date)
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    start = dt.date(date.year, 1, 1)
    days = (dt.date(date.year + 1, 1, 1) - start).days
    return date.year + (date - start).days / days


def gauss_coefficients(year: float):
    """g, h (nT) of the main field at a decimal year."""
    epochs, g, h, sg, sh = coefficients()
    if not epochs[0] <= year <= epochs[-1] + 5.0:
        raise ValueError(f"IGRF-14 covers {epochs[0]:.0f} to {epochs[-1] + 5:.0f}; got {year:.2f}")
    if year >= epochs[-1]:
        t = year - epochs[-1]
        return g[-1] + sg * t, h[-1] + sh * t
    k = min(int(np.searchsorted(epochs, year, side="right")) - 1, len(epochs) - 2)
    f = (year - epochs[k]) / (epochs[k + 1] - epochs[k])
    return g[k] + f * (g[k + 1] - g[k]), h[k] + f * (h[k + 1] - h[k])


def igrf(lon: float, lat: float, alt_km: float = 0.0, date="2025-01-01") -> dict:
    """The field at geodetic (lon, lat) in degrees, ``alt_km`` above the ellipsoid.

    Returns {"F", "H", "X", "Y", "Z" (nT; X north, Y east, Z down), "I", "D" (degrees,
    D from true north, positive east), "year", "model"}.
    """
    year = decimal_year(date)
    g, h = gauss_coefficients(year)
    nmax = g.shape[1] - 1
    # geodetic to geocentric (igrf14syn)
    colat = math.radians(90.0 - lat)
    ct, st = math.cos(colat), math.sin(colat)
    one, two = WGS84_A2 * st * st, WGS84_B2 * ct * ct
    three = one + two
    rho = math.sqrt(three)
    r = math.sqrt(alt_km * (alt_km + 2.0 * rho) + (WGS84_A2 * one + WGS84_B2 * two) / three)
    cd = (alt_km + rho) / r
    sd = (WGS84_A2 - WGS84_B2) / rho * ct * st / r
    ct, st = ct * cd - st * sd, st * cd + ct * sd
    st = max(st, 1e-10)                        # at the poles
    # Schmidt semi-normalised Legendre functions and their theta derivatives
    P = np.zeros((nmax + 1, nmax + 1))
    dP = np.zeros_like(P)
    P[0, 0] = 1.0
    for n in range(1, nmax + 1):
        k = 1.0 if n == 1 else math.sqrt((2 * n - 1) / (2 * n))
        P[n, n] = k * st * P[n - 1, n - 1]
        dP[n, n] = k * (st * dP[n - 1, n - 1] + ct * P[n - 1, n - 1])
        for m in range(n):
            a = math.sqrt(n * n - m * m)
            b = math.sqrt((n - 1) ** 2 - m * m) if n - 1 >= m else 0.0
            p2 = P[n - 2, m] if n - 2 >= m else 0.0
            d2 = dP[n - 2, m] if n - 2 >= m else 0.0
            P[n, m] = ((2 * n - 1) * ct * P[n - 1, m] - b * p2) / a
            dP[n, m] = ((2 * n - 1) * (ct * dP[n - 1, m] - st * P[n - 1, m]) - b * d2) / a
    phi = math.radians(lon)
    br = bt = bp = 0.0
    for n in range(1, nmax + 1):
        ar = (EARTH_RADIUS_KM / r) ** (n + 2)
        for m in range(n + 1):
            cm, sm = math.cos(m * phi), math.sin(m * phi)
            t = g[n, m] * cm + h[n, m] * sm
            br += (n + 1) * ar * t * P[n, m]
            bt -= ar * t * dP[n, m]
            bp += ar * m * (g[n, m] * sm - h[n, m] * cm) * P[n, m] / st
    x, y, z = -bt, bp, -br
    x, z = x * cd + z * sd, z * cd - x * sd    # back to geodetic north and down
    H = math.hypot(x, y)
    return {"F": math.hypot(H, z), "H": H, "X": x, "Y": y, "Z": z,
            "I": math.degrees(math.atan2(z, H)), "D": math.degrees(math.atan2(y, x)),
            "year": year, "model": "IGRF-14"}


def grid_convergence(x: float, y: float, crs: str) -> float:
    """The meridian convergence at (x, y) in ``crs``: the angle (degrees, clockwise) from
    true north to grid north.  East of a UTM zone's central meridian in the north it is
    positive: true north points a little west of the grid's y axis."""
    from ..io.crs import project
    lon, lat = project([x], [y], crs, "EPSG:4326")
    x2, y2 = project(lon, lat + 0.01, "EPSG:4326", crs)
    # the bearing of true north on the grid, clockwise from grid north, is minus the convergence
    return -float(math.degrees(math.atan2(x2[0] - x, y2[0] - y)))


def inducing_field(x: float, y: float, crs: str, alt_m: float = 0.0, date="2025-01-01") -> dict:
    """The IGRF at (x, y) of a projected (or geographic) CRS, ``alt_m`` above the ellipsoid.

    ``inducing_field`` = [F, I, D_grid] is what a magnetic job takes: the declination is
    measured from the CRS's grid north, D_true minus the convergence (the same for degrees).
    At the centre of the Karnataka study in UTM 43N, 2020.0: F 42,094 nT, I 19.27°,
    D -0.93° from true north, -1.35° from grid north.
    """
    from ..io.crs import project
    if crs and crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        lon, lat = project([x], [y], crs, "EPSG:4326")
        lon, lat = float(lon[0]), float(lat[0])
        gamma = grid_convergence(x, y, crs)
    else:
        lon, lat, gamma = float(x), float(y), 0.0
    f = igrf(lon, lat, alt_m / 1000.0, date)
    d_grid = f["D"] - gamma
    return {**f, "lon": lon, "lat": lat, "convergence": gamma, "D_grid": d_grid,
            "inducing_field": [round(f["F"], 1), round(f["I"], 2), round(d_grid, 2)]}
