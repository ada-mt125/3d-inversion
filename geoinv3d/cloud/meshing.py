"""Mesh design from the data: spacing detection and recommended cell sizes.

The upload page (``viz/dag_interactive.html``, "MESH DESIGN" block) implements
the same rules in JavaScript so it can show the recommendation before a job
is submitted.  Keep the two in step.

Rules
-----
* Data spacing: the grid spacing for gridded files; for station data the
  area per station, but no less than half the across-line spacing for line
  surveys (see :func:`station_spacing`).  A dataset's spacing is its finest
  file's.
* Horizontal core cell = the data spacing rounded to a "nice" number
  (1, 2, 2.5 or 5 x 10^k); vertical cell = half of it.
* Core depth and padding distance = half the survey width (the larger
  horizontal extent), the core depth rounded up to whole vertical cells.
* Cell budget: while the tensor mesh would exceed ``max_cells`` or its dense
  sensitivity matrix (n_data x n_cells x 4 bytes, float32) would exceed
  ``max_sensitivity_bytes``, step the horizontal cell up to the next nice
  number.
"""

from __future__ import annotations

import math

import numpy as np

PAD_FACTOR = 1.3
MIN_PAD_CELLS = 3
MIN_CORE_CELLS = 4
MAX_CELLS = 500_000
MAX_SENSITIVITY_BYTES = 8e9          # half the RAM of the default c5.2xlarge
SENSITIVITY_BYTES_PER_ENTRY = 4      # SimPEG stores G as float32
_NICE_MANTISSAS = (1.0, 2.0, 2.5, 5.0)


def nice_number(x: float) -> float:
    """The number of the form {1, 2, 2.5, 5} x 10^k closest to ``x`` (in log scale)."""
    if not x > 0:
        raise ValueError(f"nice_number needs a positive value, got {x}")
    k = math.floor(math.log10(x))
    candidates = [m * 10.0 ** e for e in (k - 1, k, k + 1) for m in _NICE_MANTISSAS]
    return min(candidates, key=lambda c: abs(math.log(c / x)))


def next_nice_number(x: float) -> float:
    """The smallest nice number strictly larger than ``x``."""
    k = math.floor(math.log10(x))
    for e in (k, k + 1):
        for m in _NICE_MANTISSAS:
            c = m * 10.0 ** e
            if c > x * (1 + 1e-9):
                return c
    return 10.0 ** (k + 2)


def padding_cells(h: float, pad_distance: float, factor: float = PAD_FACTOR,
                  minimum: int = MIN_PAD_CELLS) -> int:
    """Number of expanding cells (h*factor, h*factor^2, ...) that reach ``pad_distance``."""
    n, total = 0, 0.0
    while total < pad_distance and n < 200:
        n += 1
        total += h * factor ** n
    return max(minimum, n)


NN_SAMPLE = 2000   # stations whose nearest-neighbour distance is measured


def station_spacing(xy: np.ndarray) -> dict:
    """Spacing statistics of scattered stations.

    * ``area_spacing`` s: the area per station with the hull grown by s/2,
      i.e. N s^2 = A + (P/2) s + s^2 for the convex hull's area A and
      perimeter P.  Exact for a regular grid, line length / (N - 1) for
      stations on one line, and ~sqrt(A / N) for large N.
    * ``nn_spacing``: median distance to the nearest other station, over
      every k-th station (k = ceil(N / NN_SAMPLE)).
    * ``across_spacing`` = area_spacing^2 / nn_spacing: for line surveys the
      line spacing (the along-line spacing is nn_spacing).
    * ``spacing`` = max(area_spacing, across_spacing / 2): the data spacing
      used for the mesh, so dense along-line sampling does not ask for cells
      finer than half the line spacing.
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    n = len(xy)
    if n < 2:
        raise ValueError("Need at least two stations to estimate the data spacing")
    area, perimeter = 0.0, 2.0 * float(np.max(np.ptp(xy, axis=0)))
    if n >= 3:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull = ConvexHull(xy)   # in 2-D, .volume is the area and .area the perimeter
            area, perimeter = float(hull.volume), float(hull.area)
        except (QhullError, ValueError):
            pass   # collinear stations: keep the line's degenerate hull
    if perimeter <= 0:
        raise ValueError("All stations are at the same position")
    area_spacing = (perimeter / 2 + math.sqrt(perimeter ** 2 / 4 + 4 * (n - 1) * area)) \
        / (2 * (n - 1))

    from scipy.spatial import cKDTree
    step = math.ceil(n / NN_SAMPLE)
    sample = xy[::step]
    dist, _ = cKDTree(xy).query(sample, k=min(n, 8))
    nearest = [row[row > 0][0] for row in np.atleast_2d(dist) if np.any(row > 0)]
    nn_spacing = float(np.median(nearest)) if nearest else area_spacing
    across = area_spacing ** 2 / nn_spacing
    return {
        "area_spacing": area_spacing,
        "nn_spacing": nn_spacing,
        "across_spacing": across,
        "spacing": max(area_spacing, across / 2),
    }


def points_spacing(xy: np.ndarray) -> float:
    """The data spacing of scattered stations (see :func:`station_spacing`)."""
    return station_spacing(xy)["spacing"]


def tensor_shape(extent, h: float, dz: float, depth_core: float,
                 pad_distance: float) -> tuple[int, int, int]:
    """Cell counts (nx, ny, nz) of the padded tensor mesh (flat topography)."""
    xmin, xmax, ymin, ymax = extent
    n_pad = padding_cells(h, pad_distance)
    nx = max(MIN_CORE_CELLS, math.ceil((xmax - xmin) / h)) + 2 * n_pad
    ny = max(MIN_CORE_CELLS, math.ceil((ymax - ymin) / h)) + 2 * n_pad
    nz = max(MIN_CORE_CELLS, math.ceil(depth_core / dz)) + n_pad
    return nx, ny, nz


def recommend_mesh(extent, spacing: float, n_data: int, max_cells: int = MAX_CELLS,
                   max_sensitivity_bytes: float = MAX_SENSITIVITY_BYTES) -> dict:
    """Recommended tensor-mesh settings for a survey (see the module docstring).

    Args:
        extent: (xmin, xmax, ymin, ymax) of all stations.
        spacing: Data spacing in metres (finest dataset).
        n_data: Total number of observations.

    Returns:
        dict with core_cell_m, core_cell_z_m, depth_core_m, pad_distance_m,
        shape, n_cells, sensitivity_bytes, and ``coarsened_from`` (the
        spacing-based cell size) when the budget forced larger cells.
    """
    xmin, xmax, ymin, ymax = extent
    width = max(xmax - xmin, ymax - ymin, MIN_CORE_CELLS * spacing)
    h = nice_number(spacing)
    h0 = h
    pad = nice_number(0.5 * width)
    while True:
        dz = h / 2
        depth = math.ceil(0.5 * width / dz) * dz
        shape = tensor_shape(extent, h, dz, depth, pad)
        n_cells = int(np.prod(shape))
        g_bytes = float(n_data) * n_cells * SENSITIVITY_BYTES_PER_ENTRY
        if (n_cells <= max_cells and g_bytes <= max_sensitivity_bytes) or h >= width:
            break
        h = next_nice_number(h)
    return {
        "core_cell_m": h,
        "core_cell_z_m": dz,
        "depth_core_m": depth,
        "pad_distance_m": pad,
        "shape": list(shape),
        "n_cells": n_cells,
        "sensitivity_bytes": g_bytes,
        "data_spacing_m": float(spacing),
        "survey_width_m": float(width),
        "coarsened_from": h0 if h != h0 else None,
    }


# ── MT: cells from the skin depths ──
#
# The fields of a frequency f decay over a skin depth, 503 sqrt(rho / f) m.  An MT mesh needs:
# * top cells a quarter of the smallest skin depth (the highest frequency over the least
#   resistive ground the data show): with E on the ground and H in the air cell above (see
#   geoinv3d.methods.mt), the 1D scheme is then within 1 % in apparent resistivity and 1 deg
#   in phase over a halfspace, and within 4 % over a resistive layer on a 1 ohm m conductor;
# * cells growing 10 % a layer down to the depth of investigation (the Bostick depth,
#   skin depth / sqrt 2, of the lowest frequency): 1.2 doubles the error over conductors;
# * padding around and below, and air above, of twice the largest skin depth, so that the
#   secondary field has died out at the mesh's boundary (where SimPEG holds it to zero).  The
#   deep resistivity is the apparent one at the lowest frequency, raised by Niblett-Bostick
#   when it still climbs there: rho (1 + m) / (1 - m), m = d log rho_a / d log period
#   (10 over 1000 ohm m: rho_a reaches 300 at 0.3 Hz, the basement's skin depth is 29 km).
# LOGBOOK, 2026-10-07: the 1D study and the 3D checks behind these numbers.
MU0 = 4e-7 * math.pi
MT_TOP_PER_SKIN_DEPTH = 4.0
MT_TOP_LAYERS = 3
MT_Z_GROWTH = 1.1
MT_PAD_SKIN_DEPTHS = 2.0
MT_AIR_GROWTH = 1.5
MT_MAX_CELLS = 400_000            # tensor meshes; an OcTree is counted as built
MT_RHO_FLOOR = 0.1                # ohm m: no smaller resistivity is taken for the cells
MT_NB_MAX = 10.0                  # the Niblett-Bostick factor at most


def skin_depth(rho, frequency):
    """Skin depth (m) of a frequency (Hz) in a resistivity (ohm m): sqrt(2 rho / (omega mu0))."""
    return np.sqrt(2.0 * np.asarray(rho, float) / (2 * np.pi * np.asarray(frequency, float) * MU0))


def nice_floor(x: float) -> float:
    """The largest nice number ({1, 2, 2.5, 5} x 10^k) not above ``x``."""
    k = math.floor(math.log10(x))
    best = 10.0 ** (k - 1)
    for e in (k - 1, k):
        for m in _NICE_MANTISSAS:
            c = m * 10.0 ** e
            if c <= x * (1 + 1e-9):
                best = max(best, c)
    return best


def mt_rho_stats(frequencies, rho_a) -> dict | None:
    """The apparent resistivity the stations show at each frequency: its 10th, 50th and 90th
    percentiles (``rho_a``: frequencies x stations, NaN where a station has none)."""
    f = np.asarray(frequencies, float)
    r = np.asarray(rho_a, float).reshape(f.size, -1)
    r = np.where(np.isfinite(r) & (r > 0), r, np.nan)
    keep = np.isfinite(r).any(axis=1) & (f > 0)
    if not keep.any():
        return None
    f, r = f[keep], r[keep]
    order = np.argsort(f)
    pct = lambda q: np.nanpercentile(r[order], q, axis=1).tolist()   # noqa: E731
    return {"frequencies": f[order].tolist(), "p10": pct(10), "p50": pct(50), "p90": pct(90)}


def _niblett_bostick_factor(frequencies, rho_a) -> float:
    """(1 + m) / (1 - m) at the lowest frequency, m the slope of log rho_a against log period
    over the two lowest frequencies; 1 where rho_a falls, MT_NB_MAX at most."""
    f = np.asarray(frequencies, float)
    r = np.asarray(rho_a, float)
    ok = np.isfinite(r) & (r > 0) & (f > 0)
    f, r = f[ok], r[ok]
    if f.size < 2:
        return 1.0
    i = np.argsort(f)[:2]
    m = -(math.log(r[i[1]]) - math.log(r[i[0]])) / (math.log(f[i[1]]) - math.log(f[i[0]]))
    if m <= 0:
        return 1.0
    return MT_NB_MAX if m >= (MT_NB_MAX - 1) / (MT_NB_MAX + 1) else (1 + m) / (1 - m)


def mt_vertical(top: float, depth_core: float, pad: float, air: float,
                top_layers: int = MT_TOP_LAYERS, growth: float = MT_Z_GROWTH,
                pad_growth: float = PAD_FACTOR, air_growth: float = MT_AIR_GROWTH):
    """Cell thicknesses of an MT mesh from the ground down (``top_layers`` of ``top``, growing
    by ``growth`` to ``depth_core``, then by ``pad_growth`` to ``pad`` below the ground) and
    from the ground up (two of ``top``, then growing by ``air_growth`` to ``air``)."""
    down, d, h = [top] * top_layers, top * top_layers, top
    while d < depth_core and len(down) < 400:
        h *= growth
        down.append(h)
        d += h
    while d < max(pad, depth_core) and len(down) < 500:
        h *= pad_growth
        down.append(h)
        d += h
    up, a, h = [top, top], 2 * top, top
    while a < air and len(up) < 200:
        h *= air_growth
        up.append(h)
        a += h
    return np.array(down), np.array(up)


def recommend_mt_mesh(extent, spacing: float, frequencies, stats: dict | None = None,
                      rho_background: float = 100.0, max_cells: int = MT_MAX_CELLS) -> dict:
    """Recommended MT mesh (see the rules above).

    Args:
        extent: (xmin, xmax, ymin, ymax) of the stations.
        spacing: the stations' spacing (m); the horizontal cell is half of it, so that two
            cells lie between neighbours.
        frequencies: the data's frequencies (Hz).
        stats: :func:`mt_rho_stats` of the data, or None (no impedance: the background).
        rho_background: ohm m, where the data say nothing.

    Returns:
        dict with core_cell_m (horizontal), core_cell_z_m (the top cell), depth_core_m (the
        depth of investigation), pad_distance_m (around and below), air_m, z_growth,
        top_layers, margin_m (core beyond the outer stations), the skin depths and
        resistivities they come from, and the tensor mesh's shape and n_cells.
    """
    f = np.asarray(frequencies, float)
    f = f[np.isfinite(f) & (f > 0)]
    if f.size == 0:
        raise ValueError("MT mesh: no frequencies")
    f_min, f_max = float(f.min()), float(f.max())
    rho_bg = float(rho_background)
    if stats:
        p10, p50, p90 = (np.asarray(stats[k], float) for k in ("p10", "p50", "p90"))
        fs = np.asarray(stats["frequencies"], float)
        low = int(np.argmin(fs))
        rho_low = float(np.nanmin(p10))
        rho_doi, rho_deep = float(p50[low]), float(p90[low])
        rho_deep *= _niblett_bostick_factor(fs, p50)
    else:
        rho_low = rho_doi = rho_deep = rho_bg
    rho_low = max(rho_low, MT_RHO_FLOOR)
    d_min = float(skin_depth(rho_low, f_max))
    d_max = float(skin_depth(max(rho_deep, rho_low), f_min))
    top = nice_floor(d_min / MT_TOP_PER_SKIN_DEPTH)
    depth_core = float(skin_depth(max(rho_doi, rho_low), f_min)) / math.sqrt(2)
    depth_core = max(depth_core, MT_TOP_LAYERS * top * 4)
    depth_core = math.ceil(depth_core / nice_number(depth_core / 10)) * nice_number(depth_core / 10)
    pad = MT_PAD_SKIN_DEPTHS * d_max
    pad = math.ceil(pad / nice_number(pad / 10)) * nice_number(pad / 10)

    xmin, xmax, ymin, ymax = extent
    width = max(xmax - xmin, ymax - ymin)
    h = nice_number(spacing / 2)
    h0 = h
    down, up = mt_vertical(top, depth_core, pad, pad)
    while True:
        margin = max(2 * h, spacing)
        n_pad = padding_cells(h, pad, PAD_FACTOR)
        nx = max(MIN_CORE_CELLS, math.ceil((xmax - xmin + 2 * margin) / h)) + 2 * n_pad
        ny = max(MIN_CORE_CELLS, math.ceil((ymax - ymin + 2 * margin) / h)) + 2 * n_pad
        shape = (nx, ny, down.size + up.size)
        n_cells = int(np.prod(shape))
        if n_cells <= max_cells or h >= max(width, spacing):
            break
        h = next_nice_number(h)
    return {
        "core_cell_m": h, "core_cell_z_m": top, "depth_core_m": float(depth_core),
        "pad_distance_m": float(pad), "air_m": float(pad), "z_growth": MT_Z_GROWTH,
        "top_layers": MT_TOP_LAYERS, "margin_m": float(margin),
        "skin_depth_min_m": d_min, "skin_depth_max_m": d_max,
        "rho_low": rho_low, "rho_deep": rho_deep, "rho_doi": rho_doi,
        "frequency_min": f_min, "frequency_max": f_max,
        "shape": list(shape), "n_cells": n_cells, "data_spacing_m": float(spacing),
        "survey_width_m": float(width), "coarsened_from": h0 if h != h0 else None,
    }


def bostick_layers(stats: dict, max_ratio: float = 3.0):
    """A layered earth from the stations' median apparent resistivity (``stats`` of
    :func:`mt_rho_stats`) by the Niblett-Bostick transform: at each frequency the depth
    sqrt(rho_a / (omega mu0)) and the resistivity rho_a (1 + m) / (1 - m), m the slope of
    log rho_a against log period; smoothed over neighbouring frequencies and kept within
    ``max_ratio`` of the apparent resistivities.  Returns (tops, conductivities): the tops of
    the layers below the ground (the first 0, m) and their conductivities (S/m), for
    MTMethod(primary_layers=...): a primary close to the earth's layering leaves the
    secondary field the 3D part only, which dies out before the mesh's edge."""
    f = np.asarray(stats["frequencies"], float)
    r = np.asarray(stats["p50"], float)
    ok = np.isfinite(r) & (r > 0) & (f > 0)
    f, r = f[ok], r[ok]
    if f.size == 0:
        return None
    order = np.argsort(f)[::-1]                      # high frequencies (shallow) first
    f, r = f[order], r[order]
    depth = np.sqrt(r / (2 * np.pi * f * MU0))
    if f.size >= 2:
        m = np.gradient(np.log(r), np.log(1 / f))
        m = np.clip(m, -0.9, 0.9)
        rho = r * (1 + m) / (1 - m)
        if rho.size >= 3:                            # smooth: a running mean of log rho over 3
            lr = np.log(rho)
            rho = np.exp(np.convolve(np.r_[lr[0], lr, lr[-1]], np.ones(3) / 3, mode="valid"))
        rho = np.clip(rho, r.min() / max_ratio, r.max() * max_ratio)
    else:
        rho = r
    depth = np.maximum.accumulate(depth)
    keep = np.r_[True, np.diff(depth) > 0]
    depth, rho = depth[keep], rho[keep]
    tops = np.r_[0.0, np.sqrt(depth[:-1] * depth[1:])] if depth.size > 1 else np.zeros(1)
    return tops, 1.0 / rho


def _rho_a_1d(freq, rho, thick):
    """Apparent resistivity of a layered earth (rho: n layers, thick: n - 1) at ``freq``."""
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    k = np.sqrt(1j * w * MU0 / np.asarray(rho, float)[None, :])
    eta = 1j * w * MU0 / k
    z = eta[:, -1]
    for j in range(len(rho) - 2, -1, -1):
        t = np.tanh(k[:, j] * thick[j])
        z = eta[:, j] * (z + eta[:, j] * t) / (eta[:, j] + z * t)
    return np.abs(z) ** 2 / (w[:, 0] * MU0)


def smooth_1d_layers(stats: dict, n_layers: int = 30, misfit: float = 0.02):
    """A smooth layered earth fitting the stations' median apparent resistivity (``stats`` of
    :func:`mt_rho_stats`) to ``misfit`` (relative, rms) — Occam's idea: the smoothest model
    that fits, by Gauss-Newton on log resistivity in ``n_layers`` layers whose tops grow
    logarithmically from a quarter of the smallest skin depth to 1.5 times the largest,
    lowering the smoothing until the fit is reached.  Returns (tops, conductivities) as
    :func:`bostick_layers`, for MTMethod(primary_layers=...)."""
    f = np.asarray(stats["frequencies"], float)
    r = np.asarray(stats["p50"], float)
    ok = np.isfinite(r) & (r > 0) & (f > 0)
    f, r = f[ok], r[ok]
    if f.size == 0:
        return None
    if f.size < 3:
        return bostick_layers(stats)
    d_lo = float(skin_depth(r.min(), f.max())) / 4
    d_hi = float(skin_depth(r.max(), f.min())) * 1.5
    tops = np.r_[0.0, np.logspace(np.log10(d_lo), np.log10(d_hi), n_layers - 1)]
    thick = np.diff(tops)
    d = np.log(r)
    m = np.full(n_layers, np.mean(d))
    rough = np.diff(np.eye(n_layers), axis=0)
    fwd = lambda mm: np.log(_rho_a_1d(f, np.exp(mm), thick))   # noqa: E731
    for lam in np.logspace(2, -4, 13):
        for _ in range(8):
            pred = fwd(m)
            eps = 1e-4
            J = np.stack([(fwd(m + eps * e) - pred) / eps for e in np.eye(n_layers)], axis=1)
            A = J.T @ J / misfit ** 2 + lam * rough.T @ rough
            g = J.T @ (d - pred) / misfit ** 2 - lam * rough.T @ (rough @ m)
            step = np.linalg.solve(A + 1e-10 * np.eye(n_layers), g)
            m = m + np.clip(step, -2.0, 2.0)
            if np.max(np.abs(step)) < 1e-3:
                break
        if np.sqrt(np.mean((fwd(m) - d) ** 2)) <= misfit:
            break
    m = np.clip(m, np.log(r.min() / 10), np.log(r.max() * 10))
    return tops, np.exp(-m)
