"""Was a terrain correction applied to a Bouguer anomaly?  Checked from station data.

A simple Bouguer anomaly is

    BA = g_obs - gamma + 0.3086 h - 0.04193 rho h        (mGal, h in m, rho in g/cc)

and a complete one adds the terrain correction TC >= 0, which is larger where the
ground around the station is rough.  Given the observed gravity, the normal gravity
(or the latitude, for GRS80), the elevation and the anomaly of every station, the
reduction density is the least-squares rho, and the residual

    r = BA - (g_obs - gamma + 0.3086 h - 0.04193 rho h)

is only rounding noise (and a constant) for a simple Bouguer anomaly, but grows with the
local relief when a terrain correction is in it.  The local relief of a station is the
spread (max - min) of the station heights within ``radius``: the stations sample the
terrain.  The verdict compares the mean residual of the roughest and the smoothest tenth
of the stations.

The upload page runs the same check in the browser (``bouguerCheck`` in
viz/dag_interactive.html); keep the two in step.
"""

from __future__ import annotations

import numpy as np

FREE_AIR = 0.3086      # mGal per m
SLAB = 0.04193         # 2 pi G in mGal per (g/cc x m)

# column names (lower case) the check looks for in a station table
BA_NAMES = ("bouguer_an", "bouguer_anomaly", "bouguer", "ba", "cba", "sba", "bouguer_mgal", "boug")
ELEVATION_NAMES = ("elevation", "elev", "height", "h", "alt", "altitude", "elevation_m",
                   "station_elevation", "rl", "z")
OBSERVED_NAMES = ("observed_g", "observed_gravity", "obs_g", "gobs", "g_obs", "observed",
                  "g_observed", "abs_g", "absolute_gravity", "gravity", "g")
NORMAL_NAMES = ("theoretical_g", "theoretical_gravity", "normal_g", "normal_gravity", "gn",
                "g_normal", "gamma", "theoretical", "normal")
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "long", "longitude", "x")


def normal_gravity_grs80(lat_deg) -> np.ndarray:
    """Somigliana's normal gravity on the GRS80 ellipsoid (mGal)."""
    s2 = np.sin(np.radians(np.asarray(lat_deg, dtype=float))) ** 2
    return 978032.67715 * (1 + 0.0052790414 * s2 + 0.0000232718 * s2 ** 2 + 0.0000001262 * s2 ** 3)


def to_mgal(g) -> np.ndarray:
    """Absolute gravity in mGal, from m/s^2, Gal or mGal (told apart by magnitude)."""
    g = np.asarray(g, dtype=float)
    med = float(np.nanmedian(np.abs(g)))
    return g * 1e5 if med < 20 else g * 1e3 if med < 2000 else g


def check_terrain_correction(x, y, h, ba, g_obs, g_normal=None, lat=None, degrees=False,
                             radius: float = 3000.0, min_neighbours: int = 3) -> dict:
    """Reduction density and whether a terrain correction is in ``ba``.

    Args:
        x, y: station positions (m, or degrees with ``degrees=True``).
        h: elevations (m); ba: Bouguer anomaly (mGal); g_obs: observed gravity.
        g_normal: normal gravity at the stations, or ``lat`` for GRS80.
    Returns a dict with ``verdict`` ("none", "applied" or "unclear"), the fitted
    ``reduction_density``, the residual statistics and a ``message``.
    """
    from scipy.spatial import cKDTree

    x, y, h, ba = (np.asarray(v, dtype=float) for v in (x, y, h, ba))
    g_obs = to_mgal(g_obs)
    if g_normal is not None:
        gamma = to_mgal(g_normal)
    elif lat is not None:
        gamma = normal_gravity_grs80(lat)
    else:
        raise ValueError("Give the normal gravity or the latitude")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(h) & np.isfinite(ba) \
        & np.isfinite(g_obs) & np.isfinite(gamma)
    x, y, h, ba, g_obs, gamma = (v[ok] for v in (x, y, h, ba, g_obs, gamma))
    n = int(ok.sum())
    out = {"n": n, "verdict": "unclear"}
    if n < 30 or np.ptp(h) < 10:
        out["message"] = ("Too few stations, or too little elevation range, to tell the "
                          "reduction density and a terrain correction apart")
        return out
    free_air = g_obs - gamma + FREE_AIR * h
    A = np.column_stack([-SLAB * h, np.ones(n)])
    (rho, offset), *_ = np.linalg.lstsq(A, ba - free_air, rcond=None)
    res = ba - (free_air - SLAB * rho * h) - offset
    if degrees:   # local metres, enough for a 3 km neighbourhood
        y0 = np.radians(np.median(y))
        x, y = np.radians(x) * 6371e3 * np.cos(y0), np.radians(y) * 6371e3
    tree = cKDTree(np.column_stack([x, y]))
    spread = np.full(n, np.nan)
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([x, y]), radius)):
        if len(nb) > min_neighbours:
            spread[i] = np.ptp(h[nb])
    has = np.isfinite(spread)
    out.update(reduction_density=round(float(rho), 3), offset_mgal=round(float(offset), 2),
               residual_std_mgal=round(float(res.std()), 2), n_with_neighbours=int(has.sum()))
    if has.sum() < 30:
        out["message"] = (f"Reduction density {rho:.2f} g/cc; too few stations have "
                          f"neighbours within {radius / 1000:g} km to check for a terrain correction")
        return out
    r, s = res[has], spread[has]
    hi_t, lo_t = np.percentile(s, 90), np.percentile(s, 10)
    rough, smooth = r[s >= hi_t], r[s <= lo_t]
    diff = float(rough.mean() - smooth.mean())
    se = float(np.sqrt(rough.var() / len(rough) + smooth.var() / len(smooth)))
    corr = float(np.corrcoef(r, s)[0, 1]) if s.std() > 0 and r.std() > 0 else 0.0
    out.update(rough_minus_smooth_mgal=round(diff, 2), relief_corr=round(corr, 3),
               rough_relief_m=round(float(hi_t), 1))
    if diff > max(1.0, 3 * se) and corr > 0.15:
        # the terrain correction grows with height in hills and biases rho: refit on the
        # flatter half of the stations, where it is small
        idx = np.flatnonzero(has)[s <= np.median(s)]
        (rho, _), *_ = np.linalg.lstsq(A[idx], (ba - free_air)[idx], rcond=None)
        out["reduction_density"] = round(float(rho), 3)
        out["verdict"] = "applied"
        out["message"] = (f"A terrain correction appears to be applied: the anomaly is "
                          f"{diff:.1f} mGal higher at stations in rough terrain (relief over "
                          f"{hi_t:.0f} m within {radius / 1000:g} km) than in flat terrain; reduction "
                          f"density about {rho:.2f} g/cc (from the flatter half of the stations)")
    elif abs(diff) < max(0.5, 3 * se) and res.std() < 2.0:
        out["verdict"] = "none"
        out["message"] = (f"No terrain correction detected: the anomaly is a simple Bouguer "
                          f"anomaly with a reduction density of {rho:.2f} g/cc (residual "
                          f"{res.std():.1f} mGal, the same in rough and flat terrain over {n} stations)")
    else:
        out["message"] = (f"Unclear: reduction density {rho:.2f} g/cc, but the residual "
                          f"({res.std():.1f} mGal) does not settle whether a terrain correction "
                          f"was applied (rough minus flat {diff:+.1f} mGal)")
    return out


def _pick(names, lower):
    for nm in names:
        if nm in lower:
            return lower.index(nm)
    return -1


def check_table(path) -> dict | None:
    """Run the check on a station table with a header (CSV or whitespace), or None when
    the table lacks the anomaly, elevation or observed gravity."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    delim = "," if "," in head else ";" if ";" in head else None
    names = [t.strip().strip("\"'").lower() for t in (head.split(delim) if delim else head.split())]
    i_ba, i_h, i_g = _pick(BA_NAMES, names), _pick(ELEVATION_NAMES, names), _pick(OBSERVED_NAMES, names)
    if min(i_ba, i_h, i_g) < 0:
        return None
    i_n, i_lat = _pick(NORMAL_NAMES, names), _pick(LAT_NAMES, names)
    i_x, i_y = _pick(LON_NAMES, names), _pick(LAT_NAMES, names)
    if i_x < 0 or i_y < 0:
        return None
    cols = [i_x, i_y, i_h, i_ba, i_g] + ([i_n] if i_n >= 0 else [])
    data = np.genfromtxt(path, delimiter=delim, skip_header=1, usecols=cols, dtype=float,
                         invalid_raise=False)
    data = np.atleast_2d(data)
    x, y = data[:, 0], data[:, 1]
    degrees = bool(np.nanmax(np.abs(x)) <= 360 and np.nanmax(np.abs(y)) <= 90)
    if i_n < 0 and not (degrees and i_lat >= 0):
        return None
    return check_terrain_correction(
        x, y, data[:, 2], data[:, 3], data[:, 4],
        g_normal=data[:, 5] if i_n >= 0 else None, lat=y if i_n < 0 else None, degrees=degrees)
