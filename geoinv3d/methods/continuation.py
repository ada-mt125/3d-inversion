"""Upward continuation of a gridded potential field.

Data measured close to the ground carry wavelengths far shorter than the cells of a
regional mesh can produce: sampled at the mesh's spacing they alias, and the inversion
fits them with spurious near-surface structure.  Continuing the grid upwards by ``height``
multiplies each wavenumber k (rad/m) by exp(-k height): what is left is the field a
survey flown that much higher would have measured, which the mesh can model.  The data
are then inverted with their receivers at the new height.

The continuation is level-to-level: a draped survey is treated as flown on a plane, which
is a good approximation where the relief is small against the wavelengths kept.
"""

from __future__ import annotations

import numpy as np


def _padded(grid: np.ndarray, pad: int) -> np.ndarray:
    """``grid`` with ``pad`` cells on each side, fading from the edge values to their mean."""
    mean = float(np.nanmean(grid))
    out = np.pad(grid - mean, pad, mode="edge")
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(pad) / pad))       # 0 at the outside -> 1
    ny, nx = out.shape
    wy, wx = np.ones(ny), np.ones(nx)
    wy[:pad], wy[ny - pad:] = ramp, ramp[::-1]
    wx[:pad], wx[nx - pad:] = ramp, ramp[::-1]
    return out * wy[:, None] * wx[None, :] + mean


def upward_continue(grid, dx: float, dy: float, height: float, pad: int | None = None):
    """The field ``height`` metres above the plane of ``grid`` (no NaN; dx, dy in metres).

    ``pad`` cells are added on each side and faded to the mean (default: a quarter of the
    larger dimension, at least the number of cells in 3 x height), so the periodic copies
    of the FFT do not reach the grid.  ``height`` < 0 (downward continuation) is refused:
    it amplifies noise without bound.
    """
    grid = np.asarray(grid, dtype=float)
    if height < 0:
        raise ValueError("Upward continuation needs height >= 0")
    if np.isnan(grid).any():
        raise ValueError("The grid has gaps; fill them first")
    if height == 0:
        return grid.copy()
    if pad is None:
        pad = max(max(grid.shape) // 4, int(np.ceil(3 * height / min(dx, dy))))
    g = _padded(grid, pad)
    ky = 2 * np.pi * np.fft.fftfreq(g.shape[0], dy)
    kx = 2 * np.pi * np.fft.rfftfreq(g.shape[1], dx)
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = np.fft.irfft2(np.fft.rfft2(g) * np.exp(-k * height), s=g.shape)
    return out[pad:pad + grid.shape[0], pad:pad + grid.shape[1]]
