"""Gravity terrain correction from a DEM.

A simple Bouguer anomaly removes an infinite flat slab between the station and the datum.
The real ground is not flat: hills above the station pull upwards, and valleys below it are
places where the slab removed rock that is not there.  Both make the simple anomaly too
low, so the terrain correction

    TC = G rho  sum over the terrain of |vertical attraction of the rock between the
                station's level and the ground|   >= 0

is added to it (complete Bouguer anomaly = simple + TC).

The DEM is used at three resolutions around each station, on nested blocks aligned with
its grid (cell size d):

    inner    bilinear surface   the 11 x 11 cells around the station      polar integration
    near     d                  out to about 52 d                          exact prisms
    middle   5 d                out to about 250 d                         vertical lines
    far      20 d               out to ``radius``                          vertical lines

With a 90 m DEM: the surface to 0.5 km, 90 m cells to 4.7 km, 450 m to 22.5 km and 1.8 km
beyond.  Flat-topped prisms next to the station would count the step between the station
and its neighbouring cells as a slab (0.17 mGal for 30 m cells on a 10 % slope), so the
innermost block is integrated in polar coordinates over the DEM's bilinear surface, where
the integrand 1 - r / sqrt(r^2 + dh^2) stays bounded at the station.  The coarse cells
keep the mean of h^2 as well as of h: their attraction goes with the mean squared height
difference, which a mean height alone underestimates in rough terrain.

The station's height is the DEM's own (bilinear) height at the station, so a datum or
resolution difference between the survey's heights and the DEM does not appear as a slab
under the station.  The Earth's curvature is ignored (fine within ~100 km for a quantity
that is smooth at that distance and mostly removed with the regional field).
"""

from __future__ import annotations

import numpy as np

G_MGAL = 0.04193 / (2 * np.pi)     # G in mGal per (g/cc x m)

_INNER, _NEAR, _MID = 5, 10, 12    # half-widths of the nested blocks, in cells of each level
_F1, _F2 = 5, 20                   # cells of the DEM per middle / far cell
_N_AZIMUTH, _N_RADIAL = 144, 64    # polar integration of the innermost block


def _log(a, b2):
    """ln(a + sqrt(a^2 + b2)) without cancellation for a < 0."""
    r = np.sqrt(a * a + b2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(a >= 0, np.log(a + r), np.log(np.maximum(b2, 1e-300) / (r - a)))
    return out


def prism_gz(x1, x2, y1, y2, h):
    """|Vertical attraction| at the origin, per unit G rho, of prisms between z = 0 and z = h.

    x1 < x2 and y1 < y2 are the horizontal bounds relative to the station (m); ``h`` may be
    negative (rock missing below the station's level).  Returns metres: multiply by
    G_MGAL x density (g/cc) for mGal.
    """
    h = np.abs(h)
    total = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for x, sx in ((x1, -1.0), (x2, 1.0)):
            for y, sy in ((y1, -1.0), (y2, 1.0)):
                # the z = h corner minus the z = 0 corner (where the arctan term vanishes);
                # a station on an edge gives 0 x log(0), which counts as 0
                r = np.sqrt(x * x + y * y + h * h)
                atan = np.where(h > 0, h * np.arctan2(x * y, h * r), 0.0)
                top = x * _log(y, x * x + h * h) + y * _log(x, y * y + h * h) - atan
                base = x * _log(y, x * x) + y * _log(x, y * y)
                total = total + sx * sy * (np.nan_to_num(top) - np.nan_to_num(base))
    return np.abs(total)


def line_gz(r, area, h2):
    """As :func:`prism_gz` for a vertical line of cross-section ``area`` at distance ``r``;
    ``h2`` is the squared height (of a coarse cell: the mean squared height difference)."""
    return area * (1.0 / r - 1.0 / np.sqrt(r * r + h2))


def _polar_gz(z, row, col, d, hs, bounds):
    """Attraction of the terrain inside the rectangle ``bounds`` = (west, east, south,
    north), in cells from the station, integrated along rays over the bilinear surface."""
    theta = (np.arange(_N_AZIMUTH) + 0.5) * (2 * np.pi / _N_AZIMUTH)
    c, s = np.cos(theta), np.sin(theta)
    west, east, south, north = bounds
    with np.errstate(divide="ignore"):
        reach = np.minimum(np.where(c > 0, east / c, west / c),
                           np.where(s > 0, north / s, south / s))
    t = (np.arange(_N_RADIAL) + 0.5) / _N_RADIAL                 # midpoints along each ray
    r = reach[:, None] * t[None, :]                              # in cells
    dh = _bilinear(z, row + r * s[:, None], col + r * c[:, None]) - hs
    r = r * d
    f = 1.0 - r / np.sqrt(r * r + dh * dh)
    return float((f.mean(axis=1) * reach * d).sum() * (2 * np.pi / _N_AZIMUTH))


def _block_mean(z, f):
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    return z[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def _bilinear(z, row, col):
    """z at fractional (row, col) of cell centres."""
    i0 = np.clip(np.floor(row).astype(int), 0, z.shape[0] - 2)
    j0 = np.clip(np.floor(col).astype(int), 0, z.shape[1] - 2)
    fr, fc = row - i0, col - j0
    return (z[i0, j0] * (1 - fr) * (1 - fc) + z[i0, j0 + 1] * (1 - fr) * fc
            + z[i0 + 1, j0] * fr * (1 - fc) + z[i0 + 1, j0 + 1] * fr * fc)


def terrain_correction(x, y, dem_x, dem_y, dem_z, density: float = 2.67,
                       radius: float = 60000.0, return_parts: bool = False):
    """Terrain correction (mGal) at stations on the ground.

    Args:
        x, y: station coordinates (m), in the DEM's projected CRS.
        dem_x, dem_y: cell-centre coordinates of the DEM, ascending and evenly spaced (m).
        dem_z: elevations (ny, nx), ``dem_z[i, j]`` at (dem_y[i], dem_x[j]); no NaN.
        density: the Bouguer reduction density (g/cc).
        radius: half-width of the square of terrain used around each station (m).
        return_parts: also return the contributions of the four zones, (n, 4).

    Stations closer than ``radius`` to the DEM's edge get NaN.
    """
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    dem_z = np.asarray(dem_z, float)
    if np.isnan(dem_z).any():
        raise ValueError("The DEM has gaps; fill them first")
    d = float(dem_x[1] - dem_x[0])
    if d <= 0 or dem_y[1] <= dem_y[0] or abs((dem_y[1] - dem_y[0]) - d) > 1e-6 * d:
        raise ValueError("The DEM must have ascending x and y and square cells")
    x0, y0 = float(dem_x[0]) - d / 2, float(dem_y[0]) - d / 2      # outer corner of cell (0, 0)
    z0 = dem_z[:(dem_z.shape[0] // _F2) * _F2, :(dem_z.shape[1] // _F2) * _F2]
    z1, z2 = _block_mean(z0, _F1), _block_mean(z0, _F2)
    q1, q2 = _block_mean(z0 * z0, _F1), _block_mean(z0 * z0, _F2)
    d1, d2 = d * _F1, d * _F2
    n_far = int(np.ceil(radius / d2))

    # cell-centre offsets of each block, relative to the block's first cell
    def centres(n, size):
        return (np.arange(n) + 0.5) * size

    def squared(mean, mean_sq, hs):      # mean of (h - hs)^2 over a coarse cell
        return np.maximum(mean_sq - 2 * hs * mean + hs * hs, 0.0)

    out = np.full((len(x), 4), np.nan)
    n_in = 2 * _INNER + 1
    for s in range(len(x)):
        col, row = (x[s] - x0) / d, (y[s] - y0) / d            # in cells, from the outer corner
        i0, j0 = int(np.floor(row)), int(np.floor(col))
        i1, j1 = i0 // _F1, j0 // _F1
        i2, j2 = i0 // _F2, j0 // _F2
        if (i2 - n_far < 0 or j2 - n_far < 0 or i2 + n_far >= z2.shape[0]
                or j2 + n_far >= z2.shape[1]):
            continue
        hs = float(_bilinear(z0, row - 0.5, col - 0.5))

        # far: level-2 cells, without the block handed to the middle zone
        a, b = i2 - n_far, j2 - n_far
        n = 2 * n_far + 1
        yy = y0 + a * d2 + centres(n, d2)[:, None] - y[s]
        xx = x0 + b * d2 + centres(n, d2)[None, :] - x[s]
        g = line_gz(np.sqrt(xx * xx + yy * yy), d2 * d2,
                    squared(z2[a:a + n, b:b + n], q2[a:a + n, b:b + n], hs))
        g[n_far - _MID:n_far + _MID + 1, n_far - _MID:n_far + _MID + 1] = 0.0
        far = g.sum()

        # middle: level-1 cells over that block, without the block handed to the near zone
        f = _F2 // _F1
        a, b = (i2 - _MID) * f, (j2 - _MID) * f
        n = (2 * _MID + 1) * f
        yy = y0 + a * d1 + centres(n, d1)[:, None] - y[s]
        xx = x0 + b * d1 + centres(n, d1)[None, :] - x[s]
        g = line_gz(np.sqrt(xx * xx + yy * yy), d1 * d1,
                    squared(z1[a:a + n, b:b + n], q1[a:a + n, b:b + n], hs))
        ia, ja = i1 - _NEAR - a, j1 - _NEAR - b
        g[ia:ia + 2 * _NEAR + 1, ja:ja + 2 * _NEAR + 1] = 0.0
        mid = g.sum()

        # near: DEM cells over that block, without the innermost cells
        a, b = (i1 - _NEAR) * _F1, (j1 - _NEAR) * _F1
        n = (2 * _NEAR + 1) * _F1
        yc = y0 + a * d + centres(n, d)[:, None] - y[s]
        xc = x0 + b * d + centres(n, d)[None, :] - x[s]
        g = prism_gz(xc - d / 2, xc + d / 2, yc - d / 2, yc + d / 2, z0[a:a + n, b:b + n] - hs)
        ia, ja = i0 - _INNER - a, j0 - _INNER - b
        g[ia:ia + n_in, ja:ja + n_in] = 0.0
        near = g.sum()

        # inner: the innermost cells, along rays from the station to the block's sides
        inner = _polar_gz(z0, row - 0.5, col - 0.5, d, hs,
                          (j0 - _INNER - col, j0 + _INNER + 1 - col,
                           i0 - _INNER - row, i0 + _INNER + 1 - row))

        out[s] = (inner, near, mid, far)
    out *= G_MGAL * density
    total = out.sum(axis=1)
    return (total, out) if return_parts else total
