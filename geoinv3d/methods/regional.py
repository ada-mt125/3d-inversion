"""Regional-field removal before inversion.

Bouguer and magnetic anomalies carry a long-wavelength regional field from
sources deeper or wider than the model; left in, the inversion explains it
with large, spurious bodies at the mesh edges and bottom.  The usual remedy
is to fit a low-order polynomial trend surface in x, y by least squares and
invert the residual.

A ``spec`` is ``None`` / ``"none"``, ``"mean"``, ``{"method": "mean"}`` or
``{"method": "polynomial", "order": k}`` (k = 1: plane, 2: quadratic, ...).
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
    if method == "mean":
        order = 0
    elif method == "polynomial":
        order = int(spec.get("order", 1))
    else:
        raise ValueError(f"Unknown regional method '{method}' (use 'mean' or 'polynomial')")
    trend = fit_trend(xy, values, order)
    residual = values - trend
    return residual, {
        "method": method, "order": order,
        "trend_min": float(trend.min()), "trend_max": float(trend.max()),
        "data_std_before": float(values.std()), "data_std_after": float(residual.std()),
        "residual_min": float(residual.min()), "residual_max": float(residual.max()),
    }
