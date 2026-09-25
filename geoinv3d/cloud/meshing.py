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
