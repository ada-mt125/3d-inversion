"""Choosing the regularization parameter (beta, a.k.a. lambda).

SimPEG's default is the discrepancy principle: cool beta until the data misfit
reaches its expected value (chi^2 = N).  That needs trustworthy data errors.
This module adds two criteria that do not:

L-curve (Hansen 1992)
    Run the inversion at a set of fixed betas and plot log(phi_m) against
    log(phi_d).  The chosen beta sits at the corner, the point of maximum
    curvature, where lowering beta starts to buy little misfit for a lot of
    model structure.  :func:`lcurve_corner` finds it on a cubic spline through
    the points, parametrized by log(beta).

Generalized cross-validation (Wahba 1977; Golub, Heath & Wahba 1979)
    Minimize ``GCV(beta) = N ||r||^2 / (N - tr A)^2`` where ``r`` is the
    whitened residual and ``A = J (J^T J + beta R)^-1 J^T`` the influence
    (hat) matrix that maps the whitened data to the whitened prediction.
    For IRLS (sparse, L1–L2) the problem is nonlinear in the data; we use the
    quadratic model at the converged weights, i.e. ``R`` is the Hessian of
    the final majorizer, which is the usual linearization.  Cells held at a
    bound have no freedom and are left out of the trace.

:class:`FixedBetaIRLS` runs IRLS at a constant beta, which both criteria need.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import splu
from simpeg.directives import UpdateIRLS
from simpeg.regularization import BaseSparse, Sparse


def sparse_terms(reg) -> list:
    """The IRLS terms of a regularization (or combo of them), in a fixed order."""
    regs = reg.objfcts if not isinstance(reg, Sparse) else [reg]
    return [obj for r in regs if isinstance(r, Sparse) for obj in r.objfcts]


class FixedBetaIRLS(UpdateIRLS):
    """``UpdateIRLS`` that never changes beta.

    Reweighting starts after the first iteration (no L2 stage to reach a
    target misfit) and stops when phi_m changes by less than
    ``f_min_change`` between IRLS steps, or after ``max_irls_iterations``.
    Set ``invProb.beta`` before running; there must be no beta estimator.

    Args:
        irls_thresholds: Optional fixed IRLS threshold (eps) per term of
            :func:`sparse_terms`.  They are then neither re-estimated nor
            cooled, so every run minimizes the same (smoothed, convex for
            p >= 1) objective and a beta sweep traces a monotone trade-off
            curve.  Without them eps starts from a model percentile and is
            cooled, as in ``UpdateIRLS``.
    """

    def __init__(self, irls_thresholds=None, **kwargs) -> None:
        kwargs.setdefault("chifact_start", 1e300)  # start IRLS at the first endIter
        if irls_thresholds is not None:
            kwargs["irls_cooling_factor"] = 1.0
        super().__init__(**kwargs)
        self.irls_thresholds = irls_thresholds

    def start_irls(self) -> None:
        super().start_irls()
        if self.irls_thresholds is not None:
            terms = sparse_terms(self.reg)
            if len(terms) != len(self.irls_thresholds):
                raise ValueError(f"{len(self.irls_thresholds)} irls_thresholds for "
                                 f"{len(terms)} IRLS terms")
            for obj, eps in zip(terms, self.irls_thresholds):
                obj.irls_threshold = float(eps)

    def adjust_cooling_schedule(self) -> None:
        self.cooling_factor = 1.0

    def stopping_criteria(self) -> bool:
        phim_new = 0.0
        for reg in self.reg.objfcts:
            if isinstance(reg, (Sparse, BaseSparse)):
                reg.model = self.invProb.model
                phim_new += reg(reg.model)
        if self.metrics.irls_iteration_count >= self.max_irls_iterations:
            return True
        f_change = abs(self.metrics.f_old - phim_new) / (self.metrics.f_old + 1e-12)
        self.metrics.f_old = phim_new
        return bool(f_change < self.f_min_change and self.metrics.irls_iteration_count > 1)


# ── L-curve ─────────────────────────────────────────────────────────────


def lcurve_curvature(betas, phi_d, phi_m, n_eval: int = 400):
    """Signed curvature of the log-log L-curve, parametrized by log(beta).

    Returns ``(t, kappa)`` on a dense grid of ``t = log(beta)``.  With
    ``x = log phi_d`` and ``y = log phi_m`` the curve turns counter-clockwise
    at the corner, so the corner is the maximum of ``kappa``.
    """
    betas, phi_d, phi_m = (np.asarray(v, dtype=float) for v in (betas, phi_d, phi_m))
    order = np.argsort(betas)
    t = np.log(betas[order])
    x = CubicSpline(t, np.log(phi_d[order]))
    y = CubicSpline(t, np.log(phi_m[order]))
    tt = np.linspace(t[0], t[-1], n_eval)
    x1, x2, y1, y2 = x(tt, 1), x(tt, 2), y(tt, 1), y(tt, 2)
    kappa = (x1 * y2 - y1 * x2) / np.maximum(x1**2 + y1**2, 1e-300) ** 1.5
    return tt, kappa


def _distinct_points(betas, phi_d, phi_m, rel_tol: float = 1e-3):
    """Drop points that barely move along the log-log curve.

    Where the model stops changing (e.g. pinned at a bound for large beta),
    consecutive points coincide; the spline's derivatives vanish there and
    the curvature, which divides by them, shows a spurious spike.
    """
    order = np.argsort(betas)
    b, x, y = betas[order], np.log(phi_d[order]), np.log(phi_m[order])
    extent = np.hypot(np.ptp(x), np.ptp(y))
    keep = [0]
    for i in range(1, len(b)):
        if np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) > rel_tol * extent:
            keep.append(i)
    return b[keep], phi_d[order][keep], phi_m[order][keep]


def lcurve_corner(betas, phi_d, phi_m) -> float:
    """Beta at the corner (maximum curvature) of the L-curve.

    Needs at least four distinct points (see :func:`_distinct_points`).  The
    two outermost intervals are excluded, where the spline's end conditions
    dominate the curvature.
    """
    betas, phi_d, phi_m = _distinct_points(*(np.asarray(v, dtype=float)
                                             for v in (betas, phi_d, phi_m)))
    if len(betas) < 4:
        raise ValueError("The L-curve needs at least four betas with distinct points")
    tt, kappa = lcurve_curvature(betas, phi_d, phi_m)
    t_sorted = np.sort(np.log(betas))
    inner = (tt >= t_sorted[1]) & (tt <= t_sorted[-2])
    return float(np.exp(tt[inner][np.argmax(kappa[inner])]))


# ── Generalized cross-validation ────────────────────────────────────────


def influence_trace(J, reg_hessian, beta: float, free=None) -> float:
    """Exact trace of ``A = J (J^T J + beta R)^-1 J^T``.

    Uses the data-space form ``A = K (K + beta I)^-1`` with
    ``K = J R^-1 J^T`` (push-through identity), so only an N x N eigenvalue
    problem and one sparse factorization of R are needed; a diagonal R
    (smallness-only regularizations such as L1–L2) needs no factorization.

    Args:
        J: Whitened sensitivity matrix (N x M), dense.
        reg_hessian: R (M x M), half the Hessian of phi_m in SimPEG's
            convention, i.e. ``reg.deriv2(m) / 2``; must be positive definite
            on the free cells (e.g. alpha_s > 0).
        beta: Regularization parameter.
        free: Optional mask of cells that are free (not held at a bound).
    """
    J = np.asarray(J, dtype=float)
    R = sp.csc_matrix(reg_hessian)
    if free is not None:
        free = np.asarray(free, dtype=bool)
        J = J[:, free]
        R = R[free][:, free]
    if J.shape[1] == 0:
        return 0.0
    r_diag = R.diagonal()
    if abs(R - sp.diags(r_diag)).sum() == 0.0:
        RinvJt = J.T / np.maximum(r_diag, 1e-300)[:, None]
    else:
        RinvJt = splu(R).solve(np.ascontiguousarray(J.T))
    K = J @ RinvJt
    lam = np.clip(np.linalg.eigvalsh(0.5 * (K + K.T)), 0.0, None)
    return float(np.sum(lam / (lam + beta)))


def gcv_score(residual_norm2: float, trace_A: float, n_data: int) -> float:
    """``N ||r||^2 / (N - tr A)^2`` for a whitened residual ``r``."""
    dof = n_data - trace_A
    return float(n_data * residual_norm2 / max(dof, 1e-12) ** 2)


def gcv_minimum(betas, scores) -> float:
    """Beta that minimizes GCV, refined by a parabola in log(beta)."""
    betas, scores = np.asarray(betas, dtype=float), np.asarray(scores, dtype=float)
    order = np.argsort(betas)
    t, g = np.log(betas[order]), scores[order]
    i = int(np.argmin(g))
    if 0 < i < len(t) - 1:
        a, b, _ = np.polyfit(t[i - 1:i + 2], g[i - 1:i + 2], 2)
        if a > 0:
            return float(np.exp(np.clip(-b / (2 * a), t[i - 1], t[i + 1])))
    return float(np.exp(t[i]))
