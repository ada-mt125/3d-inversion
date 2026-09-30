"""Coordinates: telling degrees from metres, and projecting longitude/latitude to UTM.

The inversion works in metres.  Station tables often come in longitude/latitude
(e.g. the NGPM gravity stations) while grids are projected (the NGPM Bouguer grid is
UTM 43N), so the pipeline picks one working CRS for a job: the CRS of its gridded
data if they have one, else the ``crs`` parameter, else the UTM zone of the stations
(see :func:`working_crs`), and projects every geographic table to it.
Projections use rasterio (GDAL), which the workers have for GeoTIFFs.
"""

from __future__ import annotations

import numpy as np

GEOGRAPHIC = "EPSG:4326"


def looks_geographic(x, y) -> bool:
    """True when the coordinates can only be degrees: |x| <= 180, |y| <= 90 and the
    whole extent within 20 degrees (projected metres of a survey are never that small
    in both axes)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return False
    x, y = x[ok], y[ok]
    return bool(np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 90)
                and np.ptp(x) < 20 and np.ptp(y) < 20)


def utm_crs(lon, lat) -> str:
    """The UTM zone (WGS 84) of the median position, as "EPSG:326zz" / "EPSG:327zz"."""
    lon0, lat0 = float(np.nanmedian(lon)), float(np.nanmedian(lat))
    zone = int(np.floor((lon0 + 180.0) / 6.0)) % 60 + 1
    return f"EPSG:{32600 + zone if lat0 >= 0 else 32700 + zone}"


def project(x, y, src: str, dst: str) -> tuple[np.ndarray, np.ndarray]:
    """Transform coordinate arrays from ``src`` to ``dst`` (any CRS strings GDAL knows)."""
    from rasterio.warp import transform

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if src == dst:
        return x, y
    ok = np.isfinite(x) & np.isfinite(y)
    ox, oy = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
    if ok.any():
        tx, ty = transform(src, dst, x[ok].tolist(), y[ok].tolist())
        ox[ok], oy[ok] = tx, ty
    return ox, oy


def working_crs(grid_crs: list, points_lonlat: list, requested: str | None = None) -> str | None:
    """The CRS a job works in (see the module docstring).

    Args:
        grid_crs: CRS strings of the gridded inputs (None where unknown).
        points_lonlat: (lon, lat) arrays of the geographic station tables.
        requested: the job's ``crs`` parameter, if any.
    """
    if requested:
        return str(requested)
    known = [c for c in grid_crs if c]
    if known:
        return known[0]
    if points_lonlat:
        lon = np.concatenate([np.ravel(p[0]) for p in points_lonlat])
        lat = np.concatenate([np.ravel(p[1]) for p in points_lonlat])
        return utm_crs(lon, lat)
    return None
