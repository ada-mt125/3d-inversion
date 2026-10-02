"""Regional-field removal before inversion.

Bouguer and magnetic anomalies carry a long-wavelength regional field from
sources deeper or wider than the model; left in, the inversion explains it
with large, spurious bodies at the mesh edges and bottom.  The usual remedy
is to fit a low-order polynomial trend surface in x, y by least squares and
invert the residual.

A ``spec`` is ``None`` / ``"none"``, ``"mean"``, ``{"method": "mean"}`` or
``{"method": "polynomial", "order": k}`` (k = 1: plane, 2: quadratic, ...), or one of
the wavenumber separations of methods/enhance.py, computed on the data gridded at their
spacing and sampled back at the stations:

* ``{"method": "upward", "height_m": h}``: the regional field is the data continued
  ``h`` metres upwards (e.g. 1000 m in Yang et al. 2026);
* ``{"method": "butterworth", "cutoff_m": L, "order": n}``: the regional field holds the
  wavelengths longer than ``L`` (a Butterworth low-pass, order 4 by default);
* ``{"method": "bandpass", "short_m": a, "long_m": b}``: the residual keeps the
  wavelengths between ``a`` and ``b``; the rest is removed.

Unlike a trend surface these follow the data's own spectrum, so they need the data
on a fairly even grid; the window should be some times larger than ``L``.  Upward
continuation also weakens the broad field itself (on a test field its regional was off
by half the broad field's spread, against 5 % for the Butterworth filter), so its
residual keeps part of the regional.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def _terms(u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    """Design matrix of the monomials u^i v^j with i + j <= order."""
    cols = [u ** i * v ** (k - i) for k in range(order + 1) for i in range(k, -1, -1)]
    return np.column_stack(cols)


def fit_trend(xy: np.ndarray, values: np.ndarray, order: int) -> np.ndarray:
    """Least-squares polynomial trend surface of ``order`` at the stations."""
    if order < 0:
        raise ValueError("The trend order must be 0 or more")
    xy = np.asarray(xy, dtype=float)[:, :2]
    # Centre and scale the coordinates so the powers stay well conditioned
    centre = xy.mean(axis=0)
    scale = max(float(np.abs(xy - centre).max()), 1e-12)
    u, v = ((xy - centre) / scale).T
    design = _terms(u, v, order)
    if len(values) < design.shape[1]:
        raise ValueError(f"A trend of order {order} needs at least {design.shape[1]} data")
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coeffs


def remove_regional(xy: np.ndarray, values: np.ndarray,
                    spec: Union[None, str, dict]) -> tuple[np.ndarray, Optional[dict]]:
    """Residual data after removing the regional field, and a summary of it."""
    if spec in (None, "none", "") or (isinstance(spec, dict) and spec.get("method") in
                                       (None, "none")):
        return values, None
    if isinstance(spec, str):
        spec = {"method": spec}
    method = spec.get("method")
    values = np.asarray(values, dtype=float)
    extra = {}
    if method == "mean":
        order = 0
        label = "mean"
    elif method == "polynomial":
        order = int(spec.get("order", 1))
        label = f"trend surface of order {order}"
    elif method in SPECTRAL:
        trend, extra, label = spectral_regional(np.asarray(xy, dtype=float), values, spec)
        order = None
    else:
        raise ValueError(f"Unknown regional method '{method}' (use 'mean', 'polynomial', "
                         "'upward', 'butterworth' or 'bandpass')")
    if method not in SPECTRAL:
        trend = fit_trend(xy, values, order)
    residual = values - trend
    return residual, {
        "method": method, "order": order, "label": label, **extra,
        "trend_min": float(trend.min()), "trend_max": float(trend.max()),
        "data_std_before": float(values.std()), "data_std_after": float(residual.std()),
        "residual_min": float(residual.min()), "residual_max": float(residual.max()),
    }


SPECTRAL = ("upward", "butterworth", "bandpass")


def spectral_regional(xy: np.ndarray, values: np.ndarray, spec: dict):
    """The regional field of a wavenumber separation at the stations, its settings and a label."""
    from .enhance import bandpass, butterworth, sample, to_grid, upward
    method = spec["method"]
    xg, yg, g = to_grid(xy[:, 0], xy[:, 1], values)
    dx, dy = float(xg[1] - xg[0]), float(yg[1] - yg[0])
    if method == "upward":
        h = float(spec.get("height_m", 0))
        if not h > 0:
            raise ValueError("Upward continuation needs height_m > 0")
        reg, extra, label = upward(g, dx, dy, h), {"height_m": h}, f"upward continuation {h:g} m"
    elif method == "butterworth":
        cut, n = float(spec.get("cutoff_m", 0)), int(spec.get("order", 4))
        if not cut > 0:
            raise ValueError("A Butterworth regional needs cutoff_m > 0")
        reg, extra = butterworth(g, dx, dy, cut, n), {"cutoff_m": cut, "filter_order": n}
        label = f"Butterworth low-pass, wavelengths over {cut:g} m"
    else:
        short, long_ = float(spec.get("short_m", 0)), float(spec.get("long_m", 0))
        if not 0 < short < long_:
            raise ValueError("A band-pass needs 0 < short_m < long_m")
        from .enhance import _filled
        reg = _filled(g) - bandpass(g, dx, dy, short, long_)
        extra, label = {"short_m": short, "long_m": long_}, f"band-pass {short:g}-{long_:g} m kept"
    span = max(np.ptp(xg), np.ptp(yg))
    if method == "butterworth" and float(spec["cutoff_m"]) > 0.5 * span:
        label += " (the window is less than twice the cutoff: the regional is poorly defined)"
    extra["grid_spacing_m"] = round(dx, 3)
    return sample(xg, yg, reg, xy[:, 0], xy[:, 1]), extra, label
