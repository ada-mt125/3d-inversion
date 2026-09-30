"""Joint inversion with L2 + group-lasso regularization (Utsugi 2025), by ADMM.

Utsugi (2025, Earth Planets Space 77:146) couples a magnetic model beta and a
density model rho on the same M cells through a group lasso.  With

    f = X beta,   g = Y rho,   zeta = [beta, rho],   b = [f, g],   Z = diag(X, Y)

it minimizes

    1/2 ||b - Z zeta||^2 + lambda1 sum_k ||(beta_k, rho_k)||_2 + lambda2/2 ||zeta||^2 .

The group norm sqrt(beta_k^2 + rho_k^2) makes a cell either empty in both
models or selected in both, without asking either property to be non-zero or
the two to share a sign: a selected cell may keep rho_k = 0.  lambda2 is plain
amplitude damping (no spatial derivatives); it spreads the solution that the
group lasso alone concentrates into a few cells.  X and Y are the sensitivities
with unit columns (sensitivity weighting with gamma = 2) and the data of the
two methods are scaled to comparable amplitudes.

Generalization (:class:`GroupLassoProblem`; :class:`JointGroupLassoProblem`
is the two-model problem of the paper):

* any number P of models (physical properties), each explained by one or
  more datasets (e.g. gz and gzz data of one density model); the group of
  cell k is (zeta_1k, ..., zeta_Pk);
* nonlinear methods (MT, DC resistivity: log-conductivity relative to its
  background) by Gauss–Newton: at the current model each dataset is
  linearized, d ~ F(m_k) + J (m - m_k), the group-lasso problem of the
  linearization is solved by ADMM (warm-started), and the step to its
  solution is damped by a backtracking line search on the true objective;
* an optional cross-gradient term lambda3 C(zeta) between every pair of
  models (see :class:`CrossGradientTerm`), which asks for structurally
  similar models on top of the group lasso's co-location.

ADMM (Boyd et al. 2011) with the split s = zeta and the scaled dual u:

    zeta <- (Z^T Z + mu I)^-1 (Z^T b + mu (s + u))     one system per model
    s    <- per cell, with q = zeta - u:
            s_k = mu / (mu + lambda2) * max(1 - lambda1 / (mu ||q_k||), 0) * q_k
    u    <- u + s - zeta

until the primal residual ||s - zeta|| and the dual residual mu ||s - s_old||
are both within tolerance.  mu is the ADMM penalty: it changes how fast the
iteration gets there, not the minimizer.  With the cross-gradient the zeta
systems are coupled; they are solved model by model (Gauss–Seidel) with the
other models at their latest values, which is exact per model because the
cross-gradient is quadratic in each model when the others are fixed.

Large problems: X and Y are never formed (the sensitivity K is applied with the
weights on the fly, float32 as stored), no M x M or 2M x 2M matrix is built
when there are fewer data than cells, and the zeta systems are solved either
by a Cholesky factorization of the N x N matrix X X^T + mu I, made once and
reused for every iteration and every lambda (Woodbury:
(X^T X + mu I)^-1 = (I - X^T (X X^T + mu I)^-1 X) / mu), or matrix-free by
conjugate gradients.  See docs/group_lasso_joint.md for what follows the paper
and what is engineering.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk
from scipy.sparse.linalg import LinearOperator, cg

# Largest N x N (or M x M) float64 matrix factorized per model before
# switching to conjugate gradients
FACTOR_MAX_BYTES = 4e9
# Column blocks of the sensitivity converted to float64 at a time
CHUNK_BYTES = 256e6
DATA_SCALINGS = ("auto", "max_ratio", "std", "none")


# ── Scaling ────────────────────────────────────────────────────────────


def column_norms(K, row_scale=None, chunk_bytes: float = CHUNK_BYTES) -> np.ndarray:
    """||diag(row_scale) K[:, j]||_2 for every column, without an (N, M) temporary."""
    r2 = None if row_scale is None else np.broadcast_to(
        np.asarray(row_scale, dtype=float) ** 2, (K.shape[0],))
    if sp.issparse(K):
        sq = K.multiply(K)
        sums = sq.sum(axis=0) if r2 is None else sq.T @ r2
        return np.sqrt(np.asarray(sums, dtype=float).ravel())
    if not isinstance(K, np.ndarray):
        raise TypeError("Column norms of a matrix-free operator are not available: "
                        "pass its weights (e.g. from getJtJdiag) instead")
    n, m = K.shape
    out = np.empty(m)
    step = max(1, int(chunk_bytes // (8 * max(n, 1))))
    for j in range(0, m, step):
        block = np.asarray(K[:, j:j + step], dtype=float)
        out[j:j + step] = np.sqrt(np.einsum("ij,ij->j", block, block) if r2 is None
                                  else np.einsum("ij,ij,i->j", block, block, r2))
    return out


def sensitivity_weights(K=None, gamma: float = 2.0, norms=None, row_scale=None) -> np.ndarray:
    """w_j = (1 / ||K[:, j]||)^(gamma / 2): the physical model is w * (scaled model).

    With gamma = 2 the weighted sensitivity K diag(w) has unit columns.  Give
    ``norms`` (the column norms) for a matrix-free K; ``row_scale`` weights
    the rows first (per-datum data scaling).
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if norms is None:
        norms = column_norms(K, row_scale)
    norms = np.asarray(norms, dtype=float)
    top = norms.max() if norms.size else 0.0
    if not top > 0:
        raise ValueError("The sensitivity matrix is zero")
    return np.maximum(norms, 1e-12 * top) ** (-gamma / 2.0)


def dataset_scales(data, mode="max_ratio", stds=None, names=None, linear=None):
    """Row scale of every dataset, so that the datasets have comparable influence.

    mode:
        "max_ratio" (the paper): s_0 = 1, s_d = max|d_0| / max|d_d|.
        "std": s = 1 / std (whitening), so the misfit is chi^2.
        "none": all 1.
        "auto": "max_ratio" when every dataset is linear (``linear``), else
            "std" (the amplitudes of MT or DC data say little about their
            weight; their errors do).
        a sequence of scalars or per-datum arrays: those.

    Returns (scales, notes); notes say when a scale could not be computed.
    """
    data = [np.asarray(d, dtype=float).ravel() for d in data]
    names = list(names) if names is not None else [f"dataset {i}" for i in range(len(data))]
    stds = list(stds) if stds is not None else [None] * len(data)
    notes = []
    if isinstance(mode, (tuple, list)):
        if len(mode) != len(data):
            raise ValueError(f"{len(mode)} data scales for {len(data)} datasets")
        scales = list(mode)
    else:
        if mode == "auto":
            mode = "max_ratio" if linear is None or all(linear) else "std"
        if mode == "max_ratio":
            top0 = float(np.max(np.abs(data[0]))) if data[0].size else 0.0
            scales = [1.0]
            for d, name in zip(data[1:], names[1:]):
                top = float(np.max(np.abs(d))) if d.size else 0.0
                if top0 > 0 and top > 0:
                    scales.append(top0 / top)
                else:
                    scales.append(1.0)
                    notes.append(f"max|{names[0]}| or max|{name}| is zero: the amplitude ratio "
                                 f"is undefined, so the {name} data are not scaled (scale 1)")
        elif mode == "std":
            if any(s is None for s in stds):
                raise ValueError("data_scaling='std' needs the standard deviations of every "
                                 "dataset")
            scales = [1.0 / np.asarray(s, dtype=float) for s in stds]
        elif mode == "none":
            scales = [1.0] * len(data)
        else:
            raise ValueError(f"Unknown data_scaling {mode!r} (expected one of {DATA_SCALINGS} "
                             "or one scale per dataset)")

    def check(s, n, name):
        s = float(s) if np.ndim(s) == 0 else np.asarray(s, dtype=float)
        if np.ndim(s) and np.shape(s) != (n,):
            raise ValueError(f"The scale of {name} has shape {np.shape(s)}, expected ({n},)")
        if not np.all(np.asarray(s) > 0) or not np.all(np.isfinite(s)):
            raise ValueError(f"The scale of {name} must be positive and finite")
        return s
    return [check(s, len(d), name) for s, d, name in zip(scales, data, names)], notes


def data_scaling(f, g, mode="max_ratio", std_f=None, std_g=None):
    """Row scales (s_f, s_g) of a magnetic and a gravity dataset (see :func:`dataset_scales`).

    mode:
        "max_ratio" (the paper): s_f = 1, s_g = C = max|f| / max|g|.
        "std": s = 1 / std (whitening), so the misfit is chi^2.
        "none": both 1.
        (s_f, s_g): explicit scalars or per-datum arrays.

    Returns (s_f, s_g, notes); notes say when a scale could not be computed.
    """
    (s_f, s_g), notes = dataset_scales([f, g], mode, [std_f, std_g], ("magnetic", "gravity"))
    return s_f, s_g, notes


class WeightedOperator:
    """X = diag(S) K diag(w), applied without forming it.

    K may be a dense array (float32 stays float32: vectors are cast to its
    dtype, never the matrix to theirs), a scipy.sparse matrix, or a
    matrix-free ``scipy.sparse.linalg.LinearOperator``.  S (row scale) is a
    scalar or per-datum array, w (column weights) per cell.
    """

    def __init__(self, K, col_weights, row_scale=1.0):
        self.K = K
        self.shape = tuple(K.shape)
        self.w = np.asarray(col_weights, dtype=float)
        if self.w.shape != (self.shape[1],):
            raise ValueError(f"{self.w.size} column weights for {self.shape[1]} columns")
        self.S = float(row_scale) if np.ndim(row_scale) == 0 \
            else np.asarray(row_scale, dtype=float)
        self.dense = isinstance(K, np.ndarray)
        self.explicit = self.dense or sp.issparse(K)
        dtype = getattr(K, "dtype", np.float64)
        self._dtype = dtype if dtype in (np.float32, np.float64) else np.float64
        self.n_matvec = 0

    def matvec(self, v) -> np.ndarray:
        """X v (float64)."""
        self.n_matvec += 1
        x = (self.w * v).astype(self._dtype, copy=False)
        return self.S * np.asarray(self.K @ x, dtype=float)

    def rmatvec(self, y) -> np.ndarray:
        """X^T y (float64)."""
        self.n_matvec += 1
        z = (self.S * y).astype(self._dtype, copy=False)
        return self.w * np.asarray(self.K.T @ z, dtype=float)

    def _row_scales(self):
        return self.S if np.ndim(self.S) else np.full(self.shape[0], self.S)

    def diag_normal(self) -> np.ndarray:
        """diag(X^T X) = (w_j ||S K_j||)^2 (explicit operators)."""
        per_row = np.ndim(self.S) > 0
        return (self.w * column_norms(self.K, self.S if per_row else None)
                * (1.0 if per_row else self.S)) ** 2

    def gram(self, chunk_bytes: float = CHUNK_BYTES) -> np.ndarray:
        """Upper triangle of X X^T (N x N, float64, Fortran order).

        Accumulated over column blocks (float64 copies of one block at a time)
        with a symmetric rank-k update.
        """
        n, m = self.shape
        G = np.zeros((n, n), order="F")
        step = max(1, int(chunk_bytes // (8 * max(n, 1))))
        for j in range(0, m, step):
            block = self._columns(j, min(j + step, m))
            G = dsyrk(1.0, block, beta=1.0, c=G, trans=0, lower=0, overwrite_c=1)
        s = self._row_scales()
        G *= s[:, None]
        G *= s[None, :]
        return G

    def normal(self, chunk_bytes: float = CHUNK_BYTES) -> np.ndarray:
        """Upper triangle of X^T X (M x M, float64, Fortran order)."""
        n, m = self.shape
        H = np.zeros((m, m), order="F")
        step = max(1, int(chunk_bytes // (8 * max(m, 1))))
        s = self._row_scales()
        for i in range(0, n, step):
            rows = self._rows(i, min(i + step, n)) * s[i:i + step, None]
            H = dsyrk(1.0, rows, beta=1.0, c=H, trans=1, lower=0, overwrite_c=1)
        H *= self.w[:, None]
        H *= self.w[None, :]
        return H

    def _columns(self, j0, j1):
        block = self.K[:, j0:j1]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        return np.asfortranarray(block, dtype=float) * self.w[j0:j1]

    def _rows(self, i0, i1):
        block = self.K[i0:i1]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        return np.asfortranarray(block, dtype=float)


class StackedOperator:
    """The rows of several WeightedOperators with the same column weights.

    The operator of a model explained by several datasets: X = [X_1; X_2; ...]
    with X_d = diag(S_d) K_d diag(w).
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        if not self.blocks:
            raise ValueError("A stacked operator needs at least one block")
        m = self.blocks[0].shape[1]
        if any(b.shape[1] != m for b in self.blocks):
            raise ValueError("The stacked operators have different numbers of columns")
        self.w = self.blocks[0].w
        if any(not np.array_equal(b.w, self.w) for b in self.blocks[1:]):
            raise ValueError("The stacked operators have different column weights")
        self.shape = (sum(b.shape[0] for b in self.blocks), m)
        self.explicit = all(b.explicit for b in self.blocks)
        self.dense = all(b.dense for b in self.blocks)
        self._splits = np.cumsum([b.shape[0] for b in self.blocks])[:-1]

    @property
    def n_matvec(self) -> int:
        return max(b.n_matvec for b in self.blocks)

    def matvec(self, v) -> np.ndarray:
        return np.concatenate([b.matvec(v) for b in self.blocks])

    def rmatvec(self, y) -> np.ndarray:
        parts = np.split(np.asarray(y, dtype=float), self._splits)
        out = self.blocks[0].rmatvec(parts[0])
        for b, part in zip(self.blocks[1:], parts[1:]):
            out = out + b.rmatvec(part)
        return out

    def diag_normal(self) -> np.ndarray:
        return sum(b.diag_normal() for b in self.blocks)

    def gram(self, chunk_bytes: float = CHUNK_BYTES) -> np.ndarray:
        n, m = self.shape
        G = np.zeros((n, n), order="F")
        step = max(1, int(chunk_bytes // (8 * max(n, 1))))
        for j in range(0, m, step):
            block = np.vstack([b._columns(j, min(j + step, m)) * b._row_scales()[:, None]
                               for b in self.blocks])
            G = dsyrk(1.0, np.asfortranarray(block), beta=1.0, c=G, trans=0, lower=0,
                      overwrite_c=1)
        return G

    def normal(self, chunk_bytes: float = CHUNK_BYTES) -> np.ndarray:
        H = self.blocks[0].normal(chunk_bytes)
        for b in self.blocks[1:]:
            H += b.normal(chunk_bytes)
        return H


# ── Solvers for the zeta update: (X^T X + mu I) x = rhs ────────────────


def _pcg(apply_A, rhs, x0, precond, m: int, rtol: float, maxiter: int):
    """Preconditioned CG on an SPD operator; returns (x, iterations, info)."""
    A = LinearOperator((m, m), dtype=float, matvec=apply_A)
    M = None if precond is None else LinearOperator((m, m), dtype=float, matvec=precond)
    count = [0]

    def cb(_):
        count[0] += 1
    x, info = cg(A, rhs, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    return x, count[0], info


class _Solver:
    """Common part: ``solve_coupled`` adds a symmetric PSD term to the system.

    (X^T X + mu I + E) x = rhs by preconditioned CG, preconditioned by the
    plain solve (exact for the Cholesky solvers).
    """

    coupled_rtol = 1e-10    # the tightest tolerance of the coupled solve
    coupled_maxiter = 500

    def _precond(self):
        return self.solve

    def solve_coupled(self, rhs, x0, extra, rtol=None):
        """``rtol``: the CG tolerance (default ``coupled_rtol``); ADMM passes one that
        tightens as it converges (inexact ADMM)."""
        op, mu = self.op, self.mu
        rtol = self.coupled_rtol if rtol is None else max(rtol, self.coupled_rtol)
        x, n, info = _pcg(lambda v: op.rmatvec(op.matvec(v)) + mu * v + extra(v), rhs, x0,
                          self._precond(), op.shape[1], rtol, self.coupled_maxiter)
        self.coupled_iterations = getattr(self, "coupled_iterations", 0) + n
        if info > 0:
            warnings.warn(f"The coupled zeta update did not reach rtol={rtol} in "
                          f"{self.coupled_maxiter} CG iterations")
        return x


class _DataSpaceCholesky(_Solver):
    """Woodbury with a Cholesky factor of X X^T + mu I (fewer data than cells)."""

    name = "cholesky (data space)"

    def __init__(self, op, mu: float):
        self.op, self.mu = op, mu
        G = op.gram()
        G[np.diag_indices_from(G)] += mu
        self.factor = cho_factor(G, lower=False, overwrite_a=True, check_finite=False)

    def solve(self, rhs, x0=None):
        w = cho_solve(self.factor, self.op.matvec(rhs), check_finite=False)
        return (rhs - self.op.rmatvec(w)) / self.mu


class _ModelSpaceCholesky(_Solver):
    """Cholesky factor of X^T X + mu I (fewer cells than data)."""

    name = "cholesky (model space)"

    def __init__(self, op, mu: float):
        self.op, self.mu = op, mu
        H = op.normal()
        H[np.diag_indices_from(H)] += mu
        self.factor = cho_factor(H, lower=False, overwrite_a=True, check_finite=False)

    def solve(self, rhs, x0=None):
        return cho_solve(self.factor, rhs, check_finite=False)


class _ConjugateGradient(_Solver):
    """Matrix-free CG on X^T X + mu I, warm-started from the previous zeta.

    Jacobi preconditioner when the column norms are known (with gamma = 2 the
    diagonal is constant and it changes nothing).
    """

    name = "conjugate gradients"

    def __init__(self, op, mu: float, rtol: float = 1e-8,
                 maxiter: int = 500, diag=None):
        m = op.shape[1]
        self.op, self.mu = op, mu
        self.A = LinearOperator((m, m), dtype=float,
                                matvec=lambda v: op.rmatvec(op.matvec(v)) + mu * v)
        self._diag = diag
        self.M = None if diag is None else LinearOperator(
            (m, m), dtype=float, matvec=lambda v: v / (diag + mu))
        self.rtol, self.maxiter = rtol, maxiter
        self.coupled_rtol, self.coupled_maxiter = min(rtol, 1e-10), maxiter
        self.iterations = 0

    def _precond(self):
        return None if self._diag is None else (lambda v: v / (self._diag + self.mu))

    def solve(self, rhs, x0=None):
        count = [0]

        def cb(_):
            count[0] += 1
        x, info = cg(self.A, rhs, x0=x0, rtol=self.rtol, atol=0.0, maxiter=self.maxiter,
                     M=self.M, callback=cb)
        self.iterations += count[0]
        if info > 0:
            warnings.warn(f"CG did not reach rtol={self.rtol} in {self.maxiter} iterations")
        return x


def make_solver(op, mu: float, solver: str = "auto",
                factor_max_bytes: float = FACTOR_MAX_BYTES, cg_rtol: float = 1e-8,
                cg_maxiter: int = 500):
    """The zeta-update solver for one model.

    "auto": a Cholesky factor in the smaller of data and model space when the
    operator is explicit and the factor fits in ``factor_max_bytes``, else CG.
    """
    n, m = op.shape
    if solver == "auto":
        fits = op.explicit and min(n, m) ** 2 * 8 <= factor_max_bytes
        solver = ("cholesky" if fits else "cg")
    if solver == "cholesky":
        if not op.explicit:
            raise ValueError("A Cholesky factor needs an explicit (dense or sparse) operator")
        return _DataSpaceCholesky(op, mu) if n < m else _ModelSpaceCholesky(op, mu)
    if solver == "cg":
        diag = op.diag_normal() if op.explicit else None   # diag(X^T X) = (w_j ||S K_j||)^2
        return _ConjugateGradient(op, mu, rtol=cg_rtol, maxiter=cg_maxiter, diag=diag)
    raise ValueError(f"Unknown solver {solver!r} (expected 'auto', 'cholesky' or 'cg')")


# ── Group soft threshold ───────────────────────────────────────────────


def group_shrink_many(qs, lambda1: float, lambda2: float, mu: float, out=None,
                      coupling: str = "group", lower=None, upper=None, l2_factors=None,
                      group_weights=None):
    """Proximal step of lambda1 ||.||_group + lambda2/2 ||.||^2 at mu, for P models.

    s_pk = mu / (mu + lambda2) * max(1 - lambda1 / (mu r_k), 0) * q_pk with
    r_k = sqrt(sum_p q_pk^2): all components of a cell shrink together, and a
    cell with r_k <= lambda1 / mu (also r_k = 0) is emptied.  O(P M), vectorized.

    ``lower`` / ``upper`` (one array or None per model): the step is taken
    within these bounds, exactly (see :func:`_group_shrink_box`).
    ``l2_factors`` / ``group_weights`` (one number per model): lambda2 f_p for
    model p, and the group norm ||(g_1 s_1k, ..., g_P s_Pk)|| (the balance of the
    datasets, GroupLassoProblem.scale_models); also exact.

    ``coupling="none"`` soft-thresholds every model separately (ordinary
    elastic net on each model): a reference for comparisons, not the method.
    """
    if out is None:
        out = [np.empty_like(np.asarray(q, dtype=float)) for q in qs]
    uniform = all(v is None or np.all(np.asarray(v, dtype=float) == 1.0)
                  for v in (l2_factors, group_weights))
    if lower is not None or upper is not None or not uniform:
        return _group_shrink_box(qs, lambda1, lambda2, mu, lower, upper, out, coupling,
                                 l2_factors=l2_factors, group_weights=group_weights)
    damp = mu / (mu + lambda2)
    t = lambda1 / mu
    if coupling == "none":
        for q, o in zip(qs, out):
            o[:] = np.sign(q) * np.maximum(np.abs(q) - t, 0.0) * damp
        return out
    if coupling != "group":
        raise ValueError(f"Unknown coupling {coupling!r} (expected 'group' or 'none')")
    r = np.sqrt(sum(np.square(q) for q in qs)) if len(qs) != 2 else np.hypot(qs[0], qs[1])
    # factor = damp * max(1 - t / r, 0), with r = 0 (and r <= t) giving 0
    factor = np.subtract(r, t, out=np.zeros_like(r), where=r > t)
    np.divide(factor, r, out=factor, where=r > t)
    factor *= damp
    for q, o in zip(qs, out):
        np.multiply(factor, q, out=o)
    return out


def _box_arrays(qs, lower, upper):
    """Per-model lower and upper bounds as full arrays (None: unbounded)."""
    lo = [np.full(q.shape, -np.inf) if lower is None or lower[p] is None
          else np.broadcast_to(np.asarray(lower[p], dtype=float), q.shape)
          for p, q in enumerate(qs)]
    hi = [np.full(q.shape, np.inf) if upper is None or upper[p] is None
          else np.broadcast_to(np.asarray(upper[p], dtype=float), q.shape)
          for p, q in enumerate(qs)]
    return lo, hi


# The bounded group shrink runs per cell in parallel with numba above this many values
# (P x cells); below it, or without numba, vectorized numpy.  On the Karnataka mesh
# (2 x 335,518 values) the numpy bisection made one ADMM iteration about 1 s, most of it
# single-threaded shrink.
NUMBA_SHRINK_MIN = 50_000
_NUMBA_KERNEL = None


def _numba_shrink_kernel():
    """The per-cell bounded group shrink, compiled on first use (None without numba)."""
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        try:
            import math

            import numba
        except ImportError:
            _NUMBA_KERNEL = False
            return None

        @numba.njit(parallel=True, cache=False)
        def kernel(mq, lo, hi, a, g, lambda1, n_bisect, out):   # pragma: no cover (compiled)
            P, m = mq.shape
            for k in numba.prange(m):
                zero_ok = True
                free = 0.0
                for p in range(P):
                    lk, hk, q = lo[p, k], hi[p, k], mq[p, k]
                    if lk > 0.0 or hk < 0.0:
                        zero_ok = False
                    if not ((lk == 0.0 and q < 0.0) or (hk == 0.0 and q > 0.0)):
                        free += (q / g[p]) ** 2
                if zero_ok and math.sqrt(free) <= lambda1:
                    for p in range(P):
                        out[p, k] = 0.0
                    continue
                top = 0.0
                for p in range(P):
                    v = min(max(mq[p, k] / a[p], lo[p, k]), hi[p, k])
                    z = min(max(0.0, lo[p, k]), hi[p, k])
                    top += g[p] ** 2 * max(v * v, z * z)
                top = math.sqrt(top)
                r_lo, r_hi = 0.0, top
                for _ in range(n_bisect):
                    r = 0.5 * (r_lo + r_hi)
                    norm = 0.0
                    for p in range(P):
                        sp_ = mq[p, k] * r / (a[p] * r + lambda1 * g[p] ** 2) if lambda1 > 0.0 \
                            else mq[p, k] / a[p]
                        sp_ = min(max(sp_, lo[p, k]), hi[p, k])
                        norm += (g[p] * sp_) ** 2
                    if math.sqrt(norm) > r:
                        r_lo = r
                    else:
                        r_hi = r
                    if r_hi - r_lo <= 1e-15 * top:
                        break
                r = 0.5 * (r_lo + r_hi)
                for p in range(P):
                    sp_ = mq[p, k] * r / (a[p] * r + lambda1 * g[p] ** 2) if lambda1 > 0.0 \
                        else mq[p, k] / a[p]
                    out[p, k] = min(max(sp_, lo[p, k]), hi[p, k])
        _NUMBA_KERNEL = kernel
    return _NUMBA_KERNEL or None


def _group_shrink_box(qs, lambda1, lambda2, mu, lower, upper, out, coupling,
                      n_bisect: int = 45, l2_factors=None, group_weights=None,
                      use_numba: bool | None = None):
    """The shrink of :func:`group_shrink_many` within per-component bounds, exactly.

    Per cell it minimizes mu/2 ||s - q||^2 + lambda1 ||(g_p s_p)|| + sum_p
    lambda2 f_p / 2 s_p^2 over the box lo <= s <= hi (g = f = 1 without
    ``group_weights`` / ``l2_factors``).  With a_p = mu + lambda2 f_p the
    minimizer (unique: strictly convex) is either 0 or, for fixed
    r = ||(g_p s_p)||, the separable problem's solution

        s_p(r) = clip(mu q_p r / (a_p r + lambda1 g_p^2), lo, hi),

    so r solves r = ||(g_p s_p(r))||: one root (||g s(r)|| / r decreases), found
    by bisection on (0, sqrt(sum g_p^2 max(clip(mu q_p / a_p)^2, clip(0)^2))],
    vectorized over the cells.  0 is the minimizer when the box holds it and
    ||(mu q''_p / g_p)|| <= lambda1, where q'' drops the components that point
    out of the box at 0 (e.g. a negative susceptibility against a lower bound
    of 0).  Unbounded, unweighted components reduce to the closed form.  With equal
    weights the cells whose unconstrained group shrink lies inside the box take it
    directly (it is then the constrained minimizer), so only the cells a bound cuts are
    bisected (the bisection over all cells made the Karnataka ADMM iterations single-thread
    bound).  Large problems run the same per-cell root search in parallel with numba
    (``use_numba``; default: numba when available and P x cells >= NUMBA_SHRINK_MIN),
    stopping each cell's bisection at a relative width of 1e-15.  ``coupling="none"``:
    per component, the clipped soft threshold of lambda1 g_p |s_p| (exact in one
    dimension).
    """
    P = len(qs)
    lo, hi = _box_arrays(qs, lower, upper)
    f = [1.0] * P if l2_factors is None else [float(x) for x in l2_factors]
    g = [1.0] * P if group_weights is None else [float(x) for x in group_weights]
    a = [mu + lambda2 * fp for fp in f]
    if coupling == "group":
        size = P * np.asarray(qs[0]).size
        kernel = _numba_shrink_kernel() if (use_numba if use_numba is not None
                                            else size >= NUMBA_SHRINK_MIN) else None
        if kernel is not None:
            buf = np.empty((P, np.asarray(qs[0]).size))
            kernel(mu * np.stack([np.asarray(q, dtype=float) for q in qs]),
                   np.ascontiguousarray(np.stack(lo)), np.ascontiguousarray(np.stack(hi)),
                   np.asarray(a, dtype=float), np.asarray(g, dtype=float), float(lambda1),
                   60, buf)
            for p, o in enumerate(out):
                o[:] = buf[p]
            return out
    mq = [mu * np.asarray(q, dtype=float) for q in qs]
    if coupling == "none":
        for q, ap, gp, l, h, o in zip(mq, a, g, lo, hi, out):
            o[:] = np.clip(np.sign(q) * np.maximum(np.abs(q) - lambda1 * gp, 0.0) / ap, l, h)
        return out
    if coupling != "group":
        raise ValueError(f"Unknown coupling {coupling!r} (expected 'group' or 'none')")
    zero_ok = np.all([(l <= 0) & (h >= 0) for l, h in zip(lo, hi)], axis=0)
    free = [np.where(((l == 0) & (q < 0)) | ((h == 0) & (q > 0)), 0.0, q) / gp
            for q, gp, l, h in zip(mq, g, lo, hi)]
    empty = zero_ok & (np.sqrt(sum(np.square(x) for x in free)) <= lambda1)
    for o in out:
        o[:] = 0.0
    todo = ~empty
    if len(set(a)) == 1 and len(set(g)) == 1:
        # equal weights: where the unconstrained group shrink lies inside the box it is the
        # constrained minimizer (convex problem); only the cells a bound cuts are bisected
        rq = np.sqrt(sum(np.square(q) for q in mq))
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(rq > lambda1 * g[0], (1.0 - lambda1 * g[0] / rq) / a[0], 0.0)
        s0 = [factor * q for q in mq]
        inside = todo & (factor > 0) & np.all([(x >= l) & (x <= h)
                                               for x, l, h in zip(s0, lo, hi)], axis=0)
        for o, x in zip(out, s0):
            o[inside] = x[inside]
        todo &= ~inside
    idx = np.flatnonzero(todo)
    if not idx.size:
        return out
    mq = [q[idx] for q in mq]
    lo, hi = [l[idx] for l in lo], [h[idx] for h in hi]
    top = np.sqrt(sum(gp**2 * np.maximum(np.square(np.clip(q / ap, l, h)),
                                         np.square(np.clip(0.0, l, h)))
                      for q, ap, gp, l, h in zip(mq, a, g, lo, hi)))

    def s_of(r, p):
        return np.clip(mq[p] * r / (a[p] * r + lambda1 * g[p] ** 2), lo[p], hi[p]) \
            if lambda1 > 0 else np.clip(mq[p] / a[p], lo[p], hi[p])
    r_lo, r_hi = np.zeros_like(top), top.copy()
    for _ in range(n_bisect):
        r = 0.5 * (r_lo + r_hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.sqrt(sum(np.square(g[p] * s_of(r, p)) for p in range(P)))
        above = norm > r          # F(r) > 0: the root is further out
        r_lo = np.where(above, r, r_lo)
        r_hi = np.where(above, r_hi, r)
    r = 0.5 * (r_lo + r_hi)
    for p, o in enumerate(out):
        with np.errstate(invalid="ignore", divide="ignore"):
            o[idx] = s_of(r, p)
    return out


def group_shrink(q_beta, q_rho, lambda1: float, lambda2: float, mu: float,
                 out_beta=None, out_rho=None, coupling: str = "group"):
    """Proximal step of lambda1 ||.||_group + lambda2/2 ||.||^2 at mu: the s update.

    s_k = mu / (mu + lambda2) * max(1 - lambda1 / (mu r_k), 0) * q_k with
    r_k = sqrt(q_beta_k^2 + q_rho_k^2): both components of a cell shrink
    together, and a cell with r_k <= lambda1 / mu (also r_k = 0) is emptied.
    O(M), vectorized.  (Two models; see :func:`group_shrink_many`.)

    ``coupling="none"`` soft-thresholds beta and rho separately (ordinary
    elastic net on each model): a reference for comparisons, not the method.
    """
    out = None
    if out_beta is not None:
        out = [out_beta, out_rho]
    sb, sr = group_shrink_many([q_beta, q_rho], lambda1, lambda2, mu, out, coupling)
    return sb, sr


def group_norms(*models) -> np.ndarray:
    """||(m_1k, ..., m_Pk)|| of every cell."""
    if len(models) == 2:
        return np.hypot(models[0], models[1])
    return np.sqrt(sum(np.square(m) for m in models))


# ── Cross-gradient ─────────────────────────────────────────────────────


def largest_eigenvalue(apply, m: int, n_iter: int = 30, seed: int = 0) -> float:
    """Largest eigenvalue of a symmetric PSD operator by power iteration."""
    x = np.random.default_rng(seed).normal(size=m)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(n_iter):
        y = apply(x)
        lam = float(x @ y)
        norm = np.linalg.norm(y)
        if not norm > 0:
            return 0.0
        x = y / norm
    return lam


class CrossGradientTerm:
    """phi(u_i, u_j) = sum_c |grad u_i x grad u_j|^2 dv, discretized as SimPEG's CrossGradient.

    phi = sum_c (A g_i^2)(A g_j^2) - (A (g_i g_j))^2 with g = G u the face
    gradients and A = diag(sqrt(v)) (faces -> cells average) (Haber & Gazit
    2013).  For fixed u_j it is the quadratic u_i^T Q(u_j) u_i with

        Q(u_j) v = G^T [ (A^T A g_j^2) (G v) - g_j (A^T A (g_j (G v))) ],

    positive semidefinite (Cauchy–Schwarz, cell by cell), so the zeta update
    stays an SPD system and d phi / d u_i = 2 Q(u_j) u_i exactly.

    Args:
        mesh: discretize mesh of the models.
        active_cells: the cells the models live on (None: all).
    """

    def __init__(self, mesh, active_cells=None):
        from simpeg.regularization import RegularizationMesh

        regmesh = RegularizationMesh(mesh, active_cells=active_cells)
        self.G = regmesh.cell_gradient.tocsr()
        Av = sp.diags(np.sqrt(regmesh.vol)) @ regmesh.average_face_to_cell
        self.Av = Av.tocsr()
        self.AtA = (Av.T @ Av).tocsr()
        self.n_cells = int(regmesh.nC)

    def value(self, u_i, u_j) -> float:
        g_i, g_j = self.G @ u_i, self.G @ u_j
        Av = self.Av
        return float(np.sum((Av @ g_i**2) * (Av @ g_j**2) - (Av @ (g_i * g_j)) ** 2))

    def quadratic(self, u_other):
        """v -> Q(u_other) v."""
        g = self.G @ u_other
        d = self.AtA @ (g**2)
        G, AtA = self.G, self.AtA

        def apply(v):
            gv = G @ v
            return G.T @ (d * gv - g * (AtA @ (g * gv)))
        return apply

    def gradient(self, u, u_other) -> np.ndarray:
        """d phi(u, u_other) / d u."""
        return 2.0 * self.quadratic(u_other)(u)


# ── Datasets and the problem ───────────────────────────────────────────


@dataclass
class GroupLassoData:
    """One dataset of a group-lasso joint problem.

    Args:
        name: label (used in results, e.g. "magnetic", "gravity_2").
        data: observed data.
        model: index of the model it constrains.
        std: standard deviations (for chi^2 and data_scaling="std").
        operator: linear methods: the sensitivity K (d = K m), dense (float32
            is kept), sparse or a LinearOperator.
        simulation: nonlinear methods: a SimPEG simulation of the model (e.g.
            log-conductivity on the active cells); its ``dpred`` and ``getJ``
            (else matrix-free Jvec / Jtvec) are used at every Gauss–Newton
            iteration.
    """

    name: str
    data: Any
    model: int = 0
    std: Any = None
    operator: Any = None
    simulation: Any = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float).ravel()
        if self.std is not None:
            self.std = np.broadcast_to(np.asarray(self.std, dtype=float), self.data.shape)
        if (self.operator is None) == (self.simulation is None):
            raise ValueError(f"Dataset '{self.name}' needs either an operator (linear) or a "
                             "simulation (nonlinear)")

    @property
    def linear(self) -> bool:
        return self.operator is not None

    @property
    def n_params(self) -> int:
        if self.linear:
            return int(self.operator.shape[1])
        sim = self.simulation
        for attr in ("sigmaMap", "rhoMap", "chiMap", "model_map"):
            mapping = getattr(sim, attr, None)
            if mapping is not None:
                return int(mapping.nP)
        raise ValueError(f"Cannot tell the number of model parameters of '{self.name}'")

    def forward(self, m) -> np.ndarray:
        if self.linear:
            K = self.operator
            dtype = getattr(K, "dtype", np.float64)
            x = np.asarray(m, dtype=dtype if dtype in (np.float32, np.float64) else float)
            return np.asarray(K @ x, dtype=float)
        return np.asarray(self.simulation.dpred(np.asarray(m, dtype=float)), dtype=float)

    def jacobian(self, m):
        """d data / d m at ``m``: the operator (linear) or the simulation's J."""
        if self.linear:
            return self.operator
        sim = self.simulation
        if hasattr(sim, "getJ"):
            try:
                return np.asarray(sim.getJ(np.asarray(m, dtype=float)), dtype=float)
            except NotImplementedError:
                pass
        return sensitivity_of(sim, np.asarray(m, dtype=float))[0]


@dataclass
class ADMMState:
    """Iterates of ADMM (scaled variables, one array per model), for warm starts."""

    zeta: list
    s: list
    u: list

    @classmethod
    def zeros(cls, m: int, n_models: int = 2) -> "ADMMState":
        return cls(*([np.zeros(m) for _ in range(n_models)] for _ in range(3)))

    def copy(self) -> "ADMMState":
        return ADMMState([a.copy() for a in self.zeta], [a.copy() for a in self.s],
                         [a.copy() for a in self.u])

    # the two-model names of JointGroupLassoProblem (magnetic, gravity)
    zeta_beta = property(lambda self: self.zeta[0])
    zeta_rho = property(lambda self: self.zeta[1])
    s_beta = property(lambda self: self.s[0])
    s_rho = property(lambda self: self.s[1])
    u_beta = property(lambda self: self.u[0])
    u_rho = property(lambda self: self.u[1])


@dataclass
class GroupLassoResult:
    """Outcome of one solve.

    Models: ``models_scaled`` are the inversion variables (the s iterate,
    which has exact zeros); ``models_physical`` = reference + weights *
    scaled / (scalar data scale), in the units of the sensitivities (e.g. SI
    susceptibility, g/cc, log(S/m)).  Predicted data and residuals (one per
    dataset) are in the units of the data.  Histories (ADMM iterations, over
    all Gauss–Newton iterations of a nonlinear problem): ``misfit`` is
    1/2 ||b - Z s||^2 (scaled data, of the linearization), ``group_penalty``
    sum_k ||s_k||, ``l2_penalty`` 1/2 ||s||^2, ``cross_gradient`` C(s)
    (without the lambdas); ``objective`` = misfit + lambda1 group + lambda2 l2
    + lambda3 cross_gradient.

    ``beta_*`` / ``rho_*`` and ``*_magnetic`` / ``*_gravity`` are the first
    and second model / dataset, which :class:`JointGroupLassoProblem` makes
    the magnetic and the gravity ones.
    """

    models_scaled: list
    models_physical: list
    predicted: list
    residuals: list
    objective_history: list
    misfit_history: list
    group_penalty_history: list
    l2_penalty_history: list
    primal_residual_history: list
    dual_residual_history: list
    n_iterations: int
    converged: bool
    lambda1: float
    lambda2: float
    mu: float
    lambda1_max: float
    data_scale: tuple
    solver: str
    stopped: bool = False
    n_active: int = 0
    seconds: float = 0.0
    notes: list = field(default_factory=list)
    state: ADMMState | None = field(default=None, repr=False)
    model_names: list = field(default_factory=list)
    dataset_names: list = field(default_factory=list)
    cross_gradient: float = 0.0            # lambda3
    cross_gradient_history: list = field(default_factory=list)
    gauss_newton: list = field(default_factory=list)   # per GN iteration (nonlinear)
    final_terms: dict = field(default_factory=dict)

    @property
    def objective(self) -> float:
        return self.objective_history[-1] if self.objective_history else float("nan")

    @property
    def misfit(self) -> float:
        return self.misfit_history[-1] if self.misfit_history else float("nan")

    @property
    def group_penalty(self) -> float:
        return self.group_penalty_history[-1] if self.group_penalty_history else float("nan")

    def model(self, name: str) -> np.ndarray:
        """The physical model named ``name``."""
        return self.models_physical[self.model_names.index(name)]

    def prediction(self, name: str) -> np.ndarray:
        """The predicted data of the dataset named ``name``."""
        return self.predicted[self.dataset_names.index(name)]

    beta_scaled = property(lambda self: self.models_scaled[0])
    rho_scaled = property(lambda self: self.models_scaled[1])
    beta_physical = property(lambda self: self.models_physical[0])
    rho_physical = property(lambda self: self.models_physical[1])
    predicted_magnetic = property(lambda self: self.predicted[0])
    predicted_gravity = property(lambda self: self.predicted[1])
    residual_magnetic = property(lambda self: self.residuals[0])
    residual_gravity = property(lambda self: self.residuals[1])


class GroupLassoProblem:
    """Scaled joint problem of P models; solves it for any (lambda1, lambda2, lambda3).

    Minimizes, in the scaled variables zeta_p (physical model
    m_p = ref_p + w_p zeta_p / c_p),

        1/2 sum_d ||s_d (d_d - F_d(m_p(d)))||^2 + lambda1 sum_k ||(zeta_1k, ..., zeta_Pk)||
        + lambda2/2 sum_p ||zeta_p||^2 + lambda3 C(zeta)

    with s_d the data scales, w_p the sensitivity weights of model p (from all
    its datasets, at the reference model) and C the normalized cross-gradient
    (below).  Linear datasets (F = K m) make one ADMM solve; nonlinear ones a
    Gauss–Newton loop of ADMM solves (see the module docstring).

    Cross-gradient normalization: C = sum_{i<j} f_ij phi(u_i, u_j), where
    u_p = (w_p / median w_p) zeta_p is model p's anomaly in the scaled units
    without the depth weighting (the structure the cross-gradient compares).
    f_ij makes the curvature of the term, at a reference pair of models,
    equal to that of the data term: with the damped least-squares models
    zeta_p = (X_p^T X_p + mu I)^-1 X_p^T b_p (one solve with the factors ADMM
    uses) and lambda_max by power iteration,

        f_ij = sqrt(lambda_max(X_i^T X_i) lambda_max(X_j^T X_j)
                    / (lambda_max(H_i|j) lambda_max(H_j|i))),

    H_i|j = d^2 phi / d zeta_i^2 at the reference u_j.  So lambda3 is
    dimensionless, independent of the units of data and models, and
    lambda3 = 1 weighs the structural coupling like the data (the way
    BetaEstimate_ByEig sets beta).

    Args:
        datasets: :class:`GroupLassoData`, each with the index of its model.
        mu: ADMM penalty; the zeta-update factorizations are made for it once
            (per Gauss–Newton iteration for nonlinear models).  Default: the
            mean non-zero eigenvalue of X_p^T X_p over the models.
        model_names: one name per model (default "model_0", ...).
        references: reference model of each model (default 0), in the
            simulations' units (e.g. log(sigma_background)).
        sensitivity_weighting, gamma, model_weights: w_p = ||column||^(-gamma/2)
            of the model's stacked, row-scaled sensitivities, or none, or the
            given weights.
        cell_weights: per model, the weight R_p > 0 of each cell in the
            penalty (e.g. cell volume x depth weight, as a SimPEG
            regularization weighs its cells), or None: then w_p = 1 / R_p,
            scaled so that the columns of X_p have a mean square of 1 (as with
            gamma = 2, so lambda2 and mu keep their scale).  The group norm is
            then sum_k R_k ||(c_1 m_1k, ..., c_P m_Pk)|| up to that constant.
            model_weights, when given for a model, take precedence.
        cell_factors: per model, a factor F >= 1 on the penalty of each cell on top of the
            sensitivity weights: w_p = ||column||^(-gamma/2) / F (not rescaled, so cells
            with F = 1 keep the paper's unit columns).  With F = cell volume / smallest
            cell volume (the worker's gl_weighting="sensitivity_volume") the group norm
            counts the volume of anomalous rock instead of the number of anomalous cells:
            a padding cell of 50 core cells' volume no longer explains its share of the
            data for the price of one.
        relaxation: ADMM over-relaxation alpha in (0, 2): the s and u updates
            use alpha zeta + (1 - alpha) s_old (Boyd et al. 2011, 3.4.3); 1 is
            plain ADMM, 1.5–1.8 usually needs fewer iterations for the same
            minimizer.
        data_scaling: see :func:`dataset_scales` ("auto" by default).
        solver: "auto", "cholesky" or "cg" (see :func:`make_solver`).
        mesh, active_cells: the models' mesh (discretize) and cells, for the
            cross-gradient; None: no cross-gradient.
        bounds: (lower, upper) of each model's physical values, each a
            scalar, a per-cell array or None; None: unbounded.  They become
            bounds on the scaled variables (w and the data scale are
            positive), which the s update keeps exactly (see
            :func:`group_shrink_many`); the returned models are within them.
            Without bounds, field data with wide padding put unphysical values in
            the cells the data barely see (21.9 g/cc in the Karnataka test).

    Bounds: lambda1_max is then the smallest lambda1 whose solution is zero
    within the bounds (the components of Z^T b that point out of them at 0
    do not count), and the optimality conditions include the bounds'
    normal cone (:meth:`kkt_residuals`).  A box that does not hold the
    reference (0 in the scaled variables) has no empty cells.
    """

    def __init__(self, datasets, mu: float | None = None, *, model_names=None, references=None,
                 sensitivity_weighting: bool = True, gamma: float = 2.0, model_weights=None,
                 data_scaling="auto", solver: str = "auto",
                 factor_max_bytes: float = FACTOR_MAX_BYTES, cg_rtol: float = 1e-8,
                 cg_maxiter: int = 500, mesh=None, active_cells=None, bounds=None,
                 cell_weights=None, relaxation: float = 1.0, cell_factors=None):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("A group-lasso problem needs at least one dataset")
        self.n_models = max(d.model for d in self.datasets) + 1
        self.model_datasets = [[i for i, d in enumerate(self.datasets) if d.model == p]
                               for p in range(self.n_models)]
        if any(not ds for ds in self.model_datasets):
            raise ValueError("Every model index needs at least one dataset")
        self.model_names = list(model_names) if model_names is not None \
            else [f"model_{p}" for p in range(self.n_models)]
        if len(self.model_names) != self.n_models:
            raise ValueError(f"{len(self.model_names)} model names for {self.n_models} models")
        sizes = {d.n_params for d in self.datasets}
        if len(sizes) > 1:
            raise ValueError(f"The operators have {sorted(sizes)} cells; all models live on the "
                             "same cells")
        self.m = sizes.pop()
        for d in self.datasets:
            if d.linear and d.data.size != d.operator.shape[0]:
                raise ValueError(f"Data and operator rows do not match ({d.name})")
        self.references = [np.zeros(self.m) if references is None or references[p] is None
                           else np.broadcast_to(np.asarray(references[p], dtype=float),
                                                (self.m,)).copy()
                           for p in range(self.n_models)]
        self.linear = all(d.linear for d in self.datasets)
        self.model_linear = [all(self.datasets[i].linear for i in ds)
                             for ds in self.model_datasets]
        self.gamma = gamma
        self.notes = []
        self._solver_args = (solver, factor_max_bytes, cg_rtol, cg_maxiter)

        # data scales; a scalar scale of a model's first dataset is carried by the
        # model variable, as in the paper: the operator keeps unit columns and
        # scaled = c * physical / w, so the components of a group are in
        # comparable (data) units.  A per-datum scale (e.g. 1/std) cannot be, so it
        # weights the operator's rows, and the column weights are computed from
        # the weighted rows.
        names = [d.name for d in self.datasets]
        self.scales, notes = dataset_scales([d.data for d in self.datasets], data_scaling,
                                            [d.std for d in self.datasets], names,
                                            [d.linear for d in self.datasets])
        self.notes += notes
        for n in notes:
            warnings.warn(n)
        self.data_weights = [1.0] * self.n_models   # further factors of scale_models
        self.admm_iterations_total = 0
        self.model_factors = []
        for ds in self.model_datasets:
            s0 = self.scales[ds[0]]
            self.model_factors.append(float(s0) if np.ndim(s0) == 0 else 1.0)
        self.row_scales = [s / self.model_factors[d.model] if np.ndim(s) == 0
                           else np.asarray(s) / self.model_factors[d.model]
                           for s, d in zip(self.scales, self.datasets)]

        # Jacobians at the reference, the column weights and mu
        m0 = [ref.copy() for ref in self.references]
        self._K = [d.jacobian(m0[d.model]) for d in self.datasets]
        given = list(model_weights) if model_weights is not None else [None] * self.n_models
        cells = list(cell_weights) if cell_weights is not None else [None] * self.n_models
        factors = list(cell_factors) if cell_factors is not None else [None] * self.n_models
        if len(given) != self.n_models or len(cells) != self.n_models or len(factors) != self.n_models:
            raise ValueError(f"model_weights / cell_weights need one entry per model "
                             f"({self.n_models})")
        if not 0.0 < relaxation < 2.0:
            raise ValueError(f"relaxation must be in (0, 2), got {relaxation}")
        self.relaxation = float(relaxation)
        self.weighting = ["given" if g is not None else "cells" if c is not None
                          else f"sensitivity (gamma = {gamma:g})"
                          + (" x cell factors" if f is not None else "") if sensitivity_weighting
                          else "none" for g, c, f in zip(given, cells, factors)]
        self.weights, traces = [], []
        for p, ds in enumerate(self.model_datasets):
            explicit = all(isinstance(self._K[i], np.ndarray) or sp.issparse(self._K[i])
                           for i in ds)
            norms2 = None
            if explicit:
                norms2 = sum(column_norms(self._K[i], self.row_scales[i]
                                          if np.ndim(self.row_scales[i]) else None) ** 2
                             * (1.0 if np.ndim(self.row_scales[i]) else self.row_scales[i] ** 2)
                             for i in ds)
            elif not self.model_linear[p] and given[p] is None:
                norms2 = sum(self._jtj_diag(i, m0[p]) for i in ds)
            if given[p] is not None:
                w = np.asarray(given[p], dtype=float)
            elif cells[p] is not None:
                R = np.broadcast_to(np.asarray(cells[p], dtype=float), (self.m,))
                if not np.all(R > 0) or not np.all(np.isfinite(R)):
                    raise ValueError(f"The cell weights of '{self.model_names[p]}' must be "
                                     "positive and finite")
                if norms2 is None:
                    raise ValueError("Cell weights need the column norms of the operator; "
                                     "give model_weights for a matrix-free one")
                w = 1.0 / R
                w = w / np.sqrt(np.mean(w**2 * norms2))   # unit columns on average
            elif sensitivity_weighting:
                if norms2 is None:
                    raise ValueError("Give the weights of a matrix-free operator "
                                     "(magnetic_weights / gravity_weights / model_weights)")
                w = sensitivity_weights(gamma=gamma, norms=np.sqrt(norms2))
                if factors[p] is not None:
                    F = np.broadcast_to(np.asarray(factors[p], dtype=float), (self.m,))
                    if not np.all(F > 0) or not np.all(np.isfinite(F)):
                        raise ValueError(f"The cell factors of '{self.model_names[p]}' must be "
                                         "positive and finite")
                    w = w / F
            else:
                w = np.ones(self.m)
            self.weights.append(w)
            traces.append(None if norms2 is None else float(np.sum(w**2 * norms2)))
        self._set_bounds(bounds)
        self._build_operators()
        if mu is None:
            # the mean non-zero eigenvalue of X^T X (M / min(N, M) for unit columns):
            # ADMM converges fastest for mu near the scale of the curvature
            if any(t is None for t in traces):
                raise ValueError("mu has no default for a matrix-free operator; give it")
            mu = float(np.mean([t / min(op.shape) for t, op in zip(traces, self.ops)]))
        if not mu > 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        self.mu = float(mu)
        t0 = time.time()
        self._build_solvers(range(self.n_models))
        self.setup_seconds = time.time() - t0
        self._linearize_at(None, first=True)
        self._lambda1_max = self._zero_threshold(self.Zt_b)
        self._zt_b_ref = [z.copy() for z in self.Zt_b]
        self._ref_ops = list(self.ops)
        # cross-gradient normalization (fixed, from the reference model)
        self.cross_gradient_term = None if mesh is None else CrossGradientTerm(mesh, active_cells)
        if self.cross_gradient_term is not None and self.cross_gradient_term.n_cells != self.m:
            raise ValueError(f"The cross-gradient mesh has {self.cross_gradient_term.n_cells} "
                             f"(active) cells, the models {self.m}")
        self.w_median = [float(np.median(w)) for w in self.weights]
        self._cg_factors = None

    # -- setup helpers -------------------------------------------------------

    def _set_bounds(self, bounds):
        """Physical bounds -> bounds on the scaled variables (None when there are none)."""
        self.bounds = None
        self.box_lower = self.box_upper = None
        if bounds is None or all(b is None or (b[0] is None and b[1] is None) for b in bounds):
            return
        bounds = [None if b is None else tuple(None if v is None else np.asarray(v, dtype=float)
                                               for v in b) for b in bounds]
        if len(bounds) != self.n_models:
            raise ValueError(f"{len(bounds)} bounds for {self.n_models} models")
        self.bounds = [None if b is None else tuple(b) for b in bounds]
        lower, upper = [], []
        for p, b in enumerate(self.bounds):
            lo, hi = (None, None) if b is None else b
            ref, factor = self.references[p], self.model_factors[p] / self.weights[p]
            lo = None if lo is None else np.broadcast_to(np.asarray(lo, dtype=float), (self.m,))
            hi = None if hi is None else np.broadcast_to(np.asarray(hi, dtype=float), (self.m,))
            if lo is not None and hi is not None and np.any(lo >= hi):
                raise ValueError(f"The lower bound of '{self.model_names[p]}' is not below the "
                                 f"upper bound in {int(np.sum(lo >= hi))} cells")
            # zeta = (m - ref) c / w (a bound at the reference is exactly 0)
            lower.append(None if lo is None else (lo - ref) * factor)
            upper.append(None if hi is None else (hi - ref) * factor)
        self.box_lower, self.box_upper = lower, upper

    def _box(self):
        return (self.box_lower, self.box_upper) if self.bounds is not None else (None, None)

    def _balance_weights(self) -> dict:
        """The shrink's per-model weights after scale_models: the group norm and L2 term
        stay those of the physical model (g_p = 1 / w_p, f_p = 1 / w_p^2); {} without."""
        if all(w == 1.0 for w in self.data_weights):
            return {}
        return {"group_weights": [1.0 / w for w in self.data_weights],
                "l2_factors": [1.0 / w**2 for w in self.data_weights]}

    def _group_values(self, s_list):
        """||(g_p s_pk)|| of every cell (the group norm of the objective)."""
        return group_norms(*[s / w for s, w in zip(s_list, self.data_weights)])

    def _l2_value(self, s_list) -> float:
        return 0.5 * float(sum((s @ s) / w**2 for s, w in zip(s_list, self.data_weights)))

    def _zero_feasible(self) -> bool:
        """Whether the reference (zeta = 0) is within the bounds in every cell."""
        if self.bounds is None:
            return True
        lo, hi = _box_arrays([np.zeros(self.m)] * self.n_models, *self._box())
        return bool(all(np.all(l <= 0) and np.all(h >= 0) for l, h in zip(lo, hi)))

    def _outward_free(self, c_list):
        """``c_list`` without the components that point out of the bounds at zeta = 0."""
        if self.bounds is None:
            return c_list
        lo, hi = _box_arrays(c_list, *self._box())
        return [np.where(((l == 0) & (c < 0)) | ((h == 0) & (c > 0)), 0.0, c)
                for c, l, h in zip(c_list, lo, hi)]

    def _zero_threshold(self, zt_b) -> float:
        """max_k ||((Z^T b)_pk / g_p)||, the components pointing out of the bounds left out."""
        c = self._outward_free(list(zt_b))
        return float(np.max(group_norms(*[x * w for x, w in zip(c, self.data_weights)])))

    def _jtj_diag(self, i, m):
        sim = self.datasets[i].simulation
        r = self.row_scales[i]
        W = sp.diags(np.broadcast_to(r, (self.datasets[i].data.size,)))
        return np.asarray(sim.getJtJdiag(m, W=W), dtype=float)

    def _build_operators(self):
        ops = []
        self.dataset_ops = []
        for p, ds in enumerate(self.model_datasets):
            blocks = [WeightedOperator(self._K[i], self.weights[p], self.row_scales[i])
                      for i in ds]
            self.dataset_ops += list(zip(ds, blocks))
            ops.append(blocks[0] if len(blocks) == 1 else StackedOperator(blocks))
        self.dataset_ops = [op for _, op in sorted(self.dataset_ops, key=lambda t: t[0])]
        self.ops = ops

    def _build_solvers(self, models, damping: float = 0.0):
        """zeta-update solvers of ``models`` for mu (+ ``damping``, Levenberg–Marquardt)."""
        solver, factor_max_bytes, cg_rtol, cg_maxiter = self._solver_args
        if not hasattr(self, "solvers"):
            self.solvers = [None] * self.n_models
        for p in models:
            self.solvers[p] = make_solver(self.ops[p], self.mu + damping, solver,
                                          factor_max_bytes, cg_rtol, cg_maxiter)

    def _physical_all(self, s_list):
        m = [self.references[p] + self.weights[p] * s / self.model_factors[p]
             for p, s in enumerate(s_list)]
        if self.bounds is not None:   # the round trip through the scaling can pass a bound by 1 ulp
            m = [x if b is None else np.clip(x, -np.inf if b[0] is None else b[0],
                                             np.inf if b[1] is None else b[1])
                 for x, b in zip(m, self.bounds)]
        return m

    def _linearize_at(self, s_list, first: bool = False, damping: float = 0.0):
        """Data b of the linearization at ``s_list`` (None: the reference model).

        b_d = s_d (d_d - F_d(m)) + X_d zeta_p; for linear datasets
        b_d = s_d (d_d - K_d ref_p).  Nonlinear models get new Jacobians,
        operators and solvers (not at the first call: set up already).
        """
        if s_list is None:
            s_list = [np.zeros(self.m) for _ in range(self.n_models)]
        m = self._physical_all(s_list)
        if not first and not self.linear:
            for i, d in enumerate(self.datasets):
                if not d.linear:
                    self._K[i] = d.jacobian(m[d.model])
            self._build_operators()
            self._build_solvers([p for p in range(self.n_models) if not self.model_linear[p]],
                                damping)
        self.forward_at = []
        self.b = []
        for i, d in enumerate(self.datasets):
            p = d.model
            if d.linear:
                ref_pred = self._ref_prediction(i)
                self.b.append(self.scales[i] * (d.data - ref_pred))
                self.forward_at.append(None)
            else:
                pred = d.forward(m[p])
                self.forward_at.append(pred)
                self.b.append(self.scales[i] * (d.data - pred)
                              + self.dataset_ops[i].matvec(s_list[p]))
        self.Zt_b = []
        for p, ds in enumerate(self.model_datasets):
            z = self.dataset_ops[ds[0]].rmatvec(self.b[ds[0]])
            for i in ds[1:]:
                z = z + self.dataset_ops[i].rmatvec(self.b[i])
            self.Zt_b.append(z)
        self._lin_s = [s.copy() for s in s_list]

    def _ref_prediction(self, i):
        if not hasattr(self, "_ref_pred"):
            self._ref_pred = {}
        if i not in self._ref_pred:
            d = self.datasets[i]
            ref = self.references[d.model]
            self._ref_pred[i] = d.forward(ref) if np.any(ref) else np.zeros(d.data.size)
        return self._ref_pred[i]

    # -- quantities --------------------------------------------------------

    @property
    def solver_name(self) -> str:
        names = [s.name for s in self.solvers]
        return names[0] if len(set(names)) == 1 else " / ".join(names)

    def lambda1_max(self) -> float:
        """Smallest lambda1 whose solution is zero (the reference model): max_k ||(Z^T b)_k||."""
        return self._lambda1_max

    def scale_models(self, factors, state: "ADMMState | None" = None):
        """Weigh the data of each model by a further factor f_p > 0 (all its datasets).

        Only the data terms change: model p's misfit is multiplied by f_p^2, the group
        norm and the L2 term stay those of the same physical models — a weight per
        dataset, as SimPEG's joint data weights, for balancing the fits of the datasets
        (the worker's gl_balance).  The factor goes into the data scales s_d and the
        model's c_p together, so the operators X_p, their factorizations and mu stay as
        they are; the group norm and L2 term get the weights 1 / w_p and 1 / w_p^2 of the
        accumulated factors w_p (``data_weights``), which the shrink takes exactly.  In
        the scaled variables the same physical model is f_p zeta_p, so a warm start is
        ``state`` times f_p (returned; None without a state).  Linear datasets only.
        """
        if not self.linear:
            raise ValueError("scale_models needs linear datasets (it keeps the operators)")
        f = [float(x) for x in factors]
        if len(f) != self.n_models or not all(x > 0 and np.isfinite(x) for x in f):
            raise ValueError(f"scale_models needs {self.n_models} positive factors, got {factors}")
        for p, fp in enumerate(f):
            self.model_factors[p] *= fp
            self.data_weights[p] *= fp
            for i in self.model_datasets[p]:
                self.scales[i] = self.scales[i] * fp
        self._set_bounds(self.bounds)
        self._linearize_at(None, first=True)
        self._lambda1_max = self._zero_threshold(self.Zt_b)
        self._zt_b_ref = [z.copy() for z in self.Zt_b]
        self._cg_factors = None
        if state is None:
            return None
        return ADMMState(*([a * fp for a, fp in zip(arrs, f)]
                           for arrs in (state.zeta, state.s, state.u)))

    def to_physical(self, *s_list):
        """Models in the operators' units: reference + w * scaled / (scalar data scale).

        One array per model, or a list of them; returns a tuple.
        """
        if len(s_list) == 1 and isinstance(s_list[0], (list, tuple)):
            s_list = s_list[0]
        return tuple(self._physical_all(list(s_list)))

    def _cg_u(self, p, s):
        return (self.weights[p] / self.w_median[p]) * s

    def _cg_pair_factor(self, p, q):
        if self._cg_factors is None:
            self._cg_factors = self._cross_gradient_factors()
        return self._cg_factors[(min(p, q), max(p, q))]

    def _cross_gradient_factors(self, n_iter: int = 30) -> dict:
        """f_ij of every pair of models (see the class docstring), at the reference."""
        t = self.cross_gradient_term
        ls = [self.solvers[p].solve(z) for p, z in enumerate(self._zt_b_ref)]
        u = [self._cg_u(p, x) for p, x in enumerate(ls)]
        lam_d = [largest_eigenvalue(lambda v, op=op: op.rmatvec(op.matvec(v)), self.m, n_iter)
                 for op in self._ref_ops]

        def lam_q(p, q):
            D = self.weights[p] / self.w_median[p]
            Q = t.quadratic(u[q])
            return largest_eigenvalue(lambda v: 2.0 * D * Q(D * v), self.m, n_iter)
        factors = {}
        for p, q in combinations(range(self.n_models), 2):
            a, b = lam_q(p, q), lam_q(q, p)
            if not (a > 0 and b > 0):
                raise ValueError(f"The reference models of '{self.model_names[p]}' and "
                                 f"'{self.model_names[q]}' have no gradient: the cross-gradient "
                                 "cannot be normalized")
            factors[(p, q)] = float(np.sqrt(lam_d[p] * lam_d[q] / (a * b)))
        return factors

    def cross_gradient_value(self, s_list) -> float:
        """C(s): the normalized cross-gradient of all pairs of models (without lambda3)."""
        t = self.cross_gradient_term
        if t is None or self.n_models < 2:
            return 0.0
        u = [self._cg_u(p, s) for p, s in enumerate(s_list)]
        return float(sum(self._cg_pair_factor(p, q) * t.value(u[p], u[q])
                         for p, q in combinations(range(self.n_models), 2)))

    def cross_gradient_gradient(self, p, s_list) -> np.ndarray:
        """d C / d zeta_p."""
        t = self.cross_gradient_term
        if t is None or self.n_models < 2:
            return np.zeros(self.m)
        D = self.weights[p] / self.w_median[p]
        u_p = D * s_list[p]
        g = np.zeros(self.m)
        for q in range(self.n_models):
            if q != p:
                g += self._cg_pair_factor(p, q) * t.gradient(u_p, self._cg_u(q, s_list[q]))
        return D * g

    def _cg_hessian(self, p, s_list):
        """v -> d^2 C / d zeta_p^2 v, the others fixed (exact: C is quadratic in zeta_p)."""
        t = self.cross_gradient_term
        D = self.weights[p] / self.w_median[p]
        parts = [(self._cg_pair_factor(p, q), t.quadratic(self._cg_u(q, s_list[q])))
                 for q in range(self.n_models) if q != p]

        def apply(v):
            x = D * v
            out = np.zeros(self.m)
            for factor, Q in parts:
                out += factor * Q(x)
            return 2.0 * D * out
        return apply

    def predictions(self, s_list, exact: bool = True):
        """Predicted data of every dataset (data units).

        Linear datasets: K m.  Nonlinear ones: the forward modelling at the
        model (``exact``), or the current linearization.
        """
        m = self._physical_all(s_list)
        preds = []
        for i, d in enumerate(self.datasets):
            p = d.model
            op = self.dataset_ops[i]
            if d.linear:
                preds.append(op.matvec(s_list[p]) / self.scales[i] + self._ref_prediction(i))
            elif exact:
                same = all(np.array_equal(a, b) for a, b in zip(s_list, self._lin_s))
                preds.append(self.forward_at[i] if same else d.forward(m[p]))
            else:
                preds.append(self.forward_at[i]
                             + op.matvec(s_list[p] - self._lin_s[p]) / self.scales[i])
        return preds

    def terms_of(self, s_list, preds=None) -> dict:
        """Misfit 1/2 sum_d ||s_d (d_d - pred_d)||^2 (and per dataset), group, L2 and
        cross-gradient penalties (unweighted by the lambdas)."""
        if preds is None:
            preds = self.predictions(s_list)
        out = {"misfit": 0.0}
        for i, (d, pred) in enumerate(zip(self.datasets, preds)):
            r = self.scales[i] * (d.data - pred)
            out[f"misfit_{d.name}"] = 0.5 * float(r @ r)
            out["misfit"] += out[f"misfit_{d.name}"]
        out["group"] = float(np.sum(self._group_values(s_list)))
        out["l2"] = self._l2_value(s_list)
        out["cross_gradient"] = self.cross_gradient_value(s_list)
        return out

    def _linearized_terms(self, s_list) -> dict:
        """terms_of for the current linearization (the ADMM histories): cheap."""
        out = {"misfit": 0.0}
        for i, d in enumerate(self.datasets):
            r = self.b[i] - self.dataset_ops[i].matvec(s_list[d.model])
            out[f"misfit_{d.name}"] = 0.5 * float(r @ r)
            out["misfit"] += out[f"misfit_{d.name}"]
        out["group"] = float(np.sum(self._group_values(s_list)))
        out["l2"] = self._l2_value(s_list)
        out["cross_gradient"] = self.cross_gradient_value(s_list)
        return out

    @staticmethod
    def _objective(t, lambda1, lambda2, lambda3):
        return t["misfit"] + lambda1 * t["group"] + lambda2 * t["l2"] \
            + lambda3 * t["cross_gradient"]

    def chi2_of(self, result: "GroupLassoResult") -> dict | None:
        """chi^2 of each dataset (needs every std) and their total."""
        if any(d.std is None for d in self.datasets):
            return None
        out = {d.name: float(np.sum((r / d.std) ** 2))
               for d, r in zip(self.datasets, result.residuals)}
        out["total"] = float(sum(out.values()))
        return out

    def kkt_residuals(self, s_list, lambda1: float, lambda2: float,
                      cross_gradient: float = 0.0) -> float:
        """Largest violation of the optimality conditions, relative to lambda1.

        With c_p = Z_p^T (b - Z s) - lambda3 dC/dzeta_p (at the current
        linearization, which for a nonlinear problem is the model it was last
        solved at): a cell with s_k != 0 needs c_k = lambda1 s_k / ||s_k|| +
        lambda2 s_k; an empty cell ||c_k|| <= lambda1.  With bounds the
        difference may point out of the box where a component sits on a bound
        (the normal cone: >= 0 on an upper, <= 0 on a lower bound), and an
        empty cell's components that point out of the box at 0 do not count.
        """
        c = []
        for p, ds in enumerate(self.model_datasets):
            g = np.zeros(self.m)
            for i in ds:
                op = self.dataset_ops[i]
                g += op.rmatvec(self.b[i] - op.matvec(s_list[p]))
            if cross_gradient:
                g -= cross_gradient * self.cross_gradient_gradient(p, s_list)
            c.append(g)
        gw = [1.0 / w for w in self.data_weights]
        r = self._group_values(s_list)
        on = r > 0
        scale = max(lambda1, 1e-300)
        viol = np.zeros(self.m)
        d = [cp - lambda1 * g**2 * np.divide(sp_, r, out=np.zeros_like(r), where=on)
             - lambda2 * g**2 * sp_ for cp, sp_, g in zip(c, s_list, gw)]
        if self.bounds is not None:
            lo, hi = _box_arrays(s_list, *self._box())
            tol = [1e-10 * max(float(np.max(np.abs(x))), 1e-300) for x in s_list]
            d = [np.where((x >= h - t) & (dx > 0), 0.0, np.where((x <= l + t) & (dx < 0), 0.0, dx))
                 for x, l, h, t, dx in zip(s_list, lo, hi, tol, d)]
            c = self._outward_free(c)
        d = [x / g for x, g in zip(d, gw)]   # in the units of the group norm
        c = [x / g for x, g in zip(c, gw)]
        viol[on] = group_norms(*[x[on] for x in d]) / scale
        viol[~on] = np.maximum(group_norms(*[x[~on] for x in c]) - lambda1, 0.0) / scale
        return float(viol.max()) if viol.size else 0.0

    # -- ADMM --------------------------------------------------------------

    def _admm(self, lambda1, lambda2, lambda3, st, max_iter, tol_primal, tol_dual, tol_abs,
              history_every, callback, should_stop, coupling, hist, it0=0, prox=None):
        """ADMM on the current linearization from ``st`` (changed in place).

        ``prox``: {model: (nu, center)} adds nu/2 ||zeta_p - center||^2 (the
        Levenberg–Marquardt damping; the model's solver must be built for
        mu + nu).  Returns (iterations, converged, stopped); appends to ``hist``.
        """
        mu, m, P = self.mu, self.m, self.n_models
        sqrt_n = np.sqrt(P * m)
        rhs = np.empty(m)
        s_old = [np.empty(m) for _ in range(P)]
        q = [np.empty(m) for _ in range(P)]
        alpha = self.relaxation
        zh = [np.empty(m) for _ in range(P)] if alpha != 1.0 else None
        coupled = lambda3 > 0 and self.cross_gradient_term is not None and P >= 2
        converged = stopped = False
        it = 0
        # inexact ADMM: the coupled zeta systems are solved to a tolerance that
        # follows the primal residual (errors that shrink as ADMM converges)
        cg_rtol = 1e-3
        lam_max = self._zero_threshold(self.Zt_b)
        if coupling == "group" and lambda1 >= lam_max and prox is None \
                and self._zero_feasible():
            # the exact solution is zero (see lambda1_max; the cross-gradient has a zero
            # gradient there too); nothing to iterate
            for a in st.zeta + st.s + st.u:
                a[:] = 0.0
            return 0, True, False
        while not converged and it < max_iter:
            if should_stop is not None and should_stop():
                stopped = True
                break
            it += 1
            # 1. zeta update: one system per model (Gauss–Seidel with the cross-gradient)
            for p in range(P):
                np.add(st.s[p], st.u[p], out=rhs)
                rhs *= mu
                rhs += self.Zt_b[p]
                if prox is not None and p in prox:
                    rhs += prox[p][0] * prox[p][1]
                if coupled:   # the other models at their latest zeta
                    H = self._cg_hessian(p, st.zeta)
                    st.zeta[p] = self.solvers[p].solve_coupled(
                        rhs.copy(), st.zeta[p], lambda v, H=H: lambda3 * H(v), rtol=cg_rtol)
                else:
                    st.zeta[p] = self.solvers[p].solve(rhs, st.zeta[p])
            # 2. s update: group soft threshold of q = zeta_hat - u, zeta_hat =
            # alpha zeta + (1 - alpha) s_old (over-relaxation; zeta itself for alpha = 1)
            for p in range(P):
                s_old[p][:] = st.s[p]
            if zh is None:
                zeta_hat = st.zeta
            else:
                for p in range(P):
                    np.multiply(st.zeta[p], alpha, out=zh[p])
                    zh[p] += (1.0 - alpha) * s_old[p]
                zeta_hat = zh
            for p in range(P):
                np.subtract(zeta_hat[p], st.u[p], out=q[p])
            group_shrink_many(q, lambda1, lambda2, mu, st.s, coupling, *self._box(),
                              **self._balance_weights())
            # 3. dual update
            for p in range(P):
                st.u[p] += st.s[p] - zeta_hat[p]
            # residuals
            r_pri = float(np.sqrt(sum(np.sum((s - z) ** 2) for s, z in zip(st.s, st.zeta))))
            r_dual = mu * float(np.sqrt(sum(np.sum((s - o) ** 2) for s, o in zip(st.s, s_old))))
            norm_x = max(np.sqrt(sum(z @ z for z in st.zeta)), np.sqrt(sum(s @ s for s in st.s)))
            norm_y = mu * np.sqrt(sum(u @ u for u in st.u))
            eps_pri = sqrt_n * tol_abs + tol_primal * norm_x
            eps_dual = sqrt_n * tol_abs + tol_dual * norm_y
            hist["primal"].append(r_pri)
            hist["dual"].append(r_dual)
            converged = r_pri <= eps_pri and r_dual <= eps_dual
            cg_rtol = min(1e-3, 0.01 * max(r_pri, r_dual / mu) / max(norm_x, 1e-300))
            if history_every and (it % history_every == 0 or converged or it == max_iter):
                t = self._linearized_terms(st.s)
                obj = self._objective(t, lambda1, lambda2, lambda3)
                for k, v in (("objective", obj), ("misfit", t["misfit"]), ("group", t["group"]),
                             ("l2", t["l2"]), ("cross_gradient", t["cross_gradient"])):
                    hist[k].append(v)
                if callback is not None:
                    callback(it0 + it, {"objective": obj, **t, "primal": r_pri, "dual": r_dual,
                                        "eps_primal": eps_pri, "eps_dual": eps_dual})
        return it, converged, stopped

    def solve(self, lambda1: float, lambda2: float, *, cross_gradient: float = 0.0,
              max_iter: int = 2000, tol_primal: float = 1e-4, tol_dual: float = 1e-4,
              tol_abs: float | None = None, state: ADMMState | None = None,
              history_every: int = 1, callback=None, should_stop=None,
              coupling: str = "group", gn_max_iter: int = 20, gn_tol: float = 1e-5,
              gn_kkt_tol: float = 1e-2, gn_callback=None) -> GroupLassoResult:
        """Solve for one (lambda1, lambda2, lambda3 = ``cross_gradient``).

        ADMM stops when r_primal = ||s - zeta|| <= sqrt(P M) tol_abs +
        tol_primal max(||zeta||, ||s||) and r_dual = mu ||s - s_old|| <=
        sqrt(P M) tol_abs + tol_dual ||mu u|| (Boyd et al. 2011, section
        3.3.1), or after ``max_iter`` iterations (``converged`` False).  Never
        on the objective alone.

        Nonlinear problems repeat it in a Levenberg–Marquardt-damped
        Gauss–Newton loop (see :meth:`_gauss_newton`): linearize at the
        current model, ADMM to the damped linearized problem's minimizer
        (warm start), accept it if the true objective decreases, until the
        true problem's KKT residual is below ``gn_kkt_tol`` (relative to
        lambda1), a step changes the objective by less than ``gn_tol``
        (relative), or after ``gn_max_iter`` linearizations.  ``max_iter``
        applies to each ADMM solve.

        Args:
            tol_abs: absolute tolerance in scaled model units; default
                1e-9 max|Z^T b| (at the reference).
            state: warm start (e.g. the previous lambda1 of a sweep).
            history_every: record objective terms every so many iterations
                (each record costs one product with every operator).
            callback: ``callback(iteration, record)`` after each record.
            should_stop: ``() -> bool`` asked every iteration; True ends the
                solve with the current iterate (``stopped``).
            gn_callback: ``gn_callback(k, record)`` after each Gauss–Newton step.
        """
        if lambda1 < 0 or lambda2 < 0 or cross_gradient < 0:
            raise ValueError("lambda1, lambda2 and cross_gradient must be >= 0")
        if cross_gradient > 0 and self.cross_gradient_term is None:
            raise ValueError("The cross-gradient needs the models' mesh (give mesh= to the "
                             "problem)")
        t0 = time.time()
        if tol_abs is None:
            tol_abs = 1e-9 * max(float(np.max(np.abs(np.concatenate(self._zt_b_ref)))),
                                 1e-300)
        st = ADMMState.zeros(self.m, self.n_models) if state is None else state.copy()
        hist = {k: [] for k in ("objective", "misfit", "group", "l2", "cross_gradient",
                                "primal", "dual")}
        args = (max_iter, tol_primal, tol_dual, tol_abs, history_every, callback, should_stop,
                coupling, hist)
        gn = []
        if self.linear:
            it, converged, stopped = self._admm(lambda1, lambda2, cross_gradient, st, *args)
        else:
            it, converged, stopped = self._gauss_newton(
                lambda1, lambda2, cross_gradient, st, args, gn_max_iter, gn_tol, gn_kkt_tol, gn,
                gn_callback)
        self.admm_iterations_total += it     # every solve of this problem (sweeps, searches, balance)
        return self._result(st, lambda1, lambda2, cross_gradient, hist, it, converged, stopped,
                            time.time() - t0, gn)

    def _gauss_newton(self, lambda1, lambda2, lambda3, st, args, gn_max_iter, gn_tol,
                      gn_kkt_tol, gn, gn_callback):
        """Levenberg–Marquardt-damped Gauss–Newton on the nonlinear models.

        Each iteration solves the problem linearized at s_k plus nu/2 ||zeta - s_k||^2
        on the nonlinear models (their zeta systems get mu + nu; the linear
        models, whose misfit is exact, are not damped).  The step is accepted
        when the true objective decreases; nu is lowered after a step whose
        decrease matches the linearization's prediction (rho > 0.75), raised
        after a poor one (rho < 0.25) and after a rejected one (retried).
        The damping vanishes at a fixed point, which is a stationary point of
        the true problem.  Converged when the KKT residual of the true problem
        (:meth:`kkt_residuals` at the current linearization, which is exact
        at the current model) is below ``gn_kkt_tol``, or when a step with at
        most moderate damping (nu <= mu) changes the objective by less than
        ``gn_tol`` (relative): in the poorly resolved directions of an
        ill-posed problem the models then still move slowly, but the
        objective no longer does.  Stopped without convergence when no damped
        step decreases the objective, or after ``gn_max_iter`` steps.
        ``self.gn_stop`` says which.
        """
        total, converged, stopped = 0, False, False
        hist = args[-1]
        nonlinear = [p for p in range(self.n_models) if not self.model_linear[p]]
        nu = 0.0
        self.kkt_history = []
        for k in range(gn_max_iter + 1):
            if not all(np.array_equal(a, b) for a, b in zip(st.s, self._lin_s)):
                self._linearize_at(st.s, damping=nu)
            kkt = self.kkt_residuals(st.s, lambda1, lambda2, lambda3)
            self.kkt_history.append(kkt)
            if gn:
                gn[-1]["kkt"] = kkt
            if k > 0 and kkt <= gn_kkt_tol:
                converged, self.gn_stop = True, "kkt"
                break
            if k == gn_max_iter:
                self.gn_stop = "max_iter"
                break
            s0 = [s.copy() for s in st.s]
            t_old = self.terms_of(s0)
            obj0 = self._objective(t_old, lambda1, lambda2, lambda3)
            accepted = False
            for _ in range(8):
                prox = {p: (nu, s0[p]) for p in nonlinear} if nu > 0 else None
                trial = st.copy()
                it, admm_ok, stopped = self._admm(lambda1, lambda2, lambda3, trial, *args[:-1],
                                                  hist, it0=total, prox=prox)
                total += it
                if stopped:
                    break
                t_lin = self._linearized_terms(trial.s)
                t_new = self.terms_of(trial.s)
                obj = self._objective(t_new, lambda1, lambda2, lambda3)
                predicted = obj0 - self._objective(t_lin, lambda1, lambda2, lambda3)
                actual = obj0 - obj
                rho = actual / predicted if predicted > 0 else (1.0 if actual >= 0 else -1.0)
                if actual >= 0:
                    accepted = True
                    break
                nu = max(2.0 * nu, self.mu)   # rejected: damp more and retry
                self._build_solvers(nonlinear, damping=nu)
            if stopped:
                st.zeta, st.s, st.u = trial.zeta, trial.s, trial.u
                break
            if accepted:
                st.zeta, st.s, st.u = trial.zeta, trial.s, trial.u
            else:
                obj, t_new, rho = obj0, t_old, 0.0
            nu_used = nu
            record = {"iteration": k + 1, "objective": obj, "misfit": t_new["misfit"],
                      "damping": nu, "rho": float(rho), "accepted": accepted,
                      "admm_iterations": it, "admm_converged": bool(admm_ok)}
            gn.append(record)
            if gn_callback is not None:
                gn_callback(k + 1, record)
            if rho > 0.75:
                nu = nu / 3.0 if nu > 1e-3 * self.mu else 0.0
            elif rho < 0.25:
                nu = max(2.0 * nu, self.mu)
            change = abs(obj0 - obj) / max(abs(obj0), 1e-300)
            if not accepted:
                self.gn_stop = "no_decrease"
                break
            if change <= gn_tol and nu_used <= self.mu:
                converged, self.gn_stop = True, "objective"
                break
        if not all(np.array_equal(a, b) for a, b in zip(st.s, self._lin_s)):
            self._linearize_at(st.s)   # predictions, residuals and KKT at the final model
        return total, converged, stopped

    def _result(self, st, lambda1, lambda2, lambda3, hist, it, converged, stopped, seconds, gn):
        phys = self._physical_all(st.s)
        preds = self.predictions(st.s)
        t = self.terms_of(st.s, preds)
        if not hist["objective"] or it == 0 or not self.linear:
            # the true objective of the final model (nonlinear: not the linearization's)
            for k, v in (("objective", self._objective(t, lambda1, lambda2, lambda3)),
                         ("misfit", t["misfit"]), ("group", t["group"]), ("l2", t["l2"]),
                         ("cross_gradient", t["cross_gradient"])):
                if self.linear and hist[k] and it > 0:
                    continue
                hist[k].append(v)
        return GroupLassoResult(
            models_scaled=[s.copy() for s in st.s], models_physical=phys, predicted=preds,
            residuals=[d.data - p for d, p in zip(self.datasets, preds)],
            objective_history=hist["objective"], misfit_history=hist["misfit"],
            group_penalty_history=hist["group"], l2_penalty_history=hist["l2"],
            primal_residual_history=hist["primal"], dual_residual_history=hist["dual"],
            n_iterations=it, converged=bool(converged), lambda1=float(lambda1),
            lambda2=float(lambda2), mu=self.mu, lambda1_max=self._lambda1_max,
            data_scale=tuple(self.scales), solver=self.solver_name, stopped=stopped,
            n_active=int(np.count_nonzero(group_norms(*st.s))),
            seconds=seconds, notes=list(self.notes), state=st,
            model_names=list(self.model_names), dataset_names=[d.name for d in self.datasets],
            cross_gradient=float(lambda3), cross_gradient_history=hist["cross_gradient"],
            gauss_newton=gn, final_terms=t)

    # -- L-curve over lambda1 ----------------------------------------------

    def lcurve(self, lambda2: float, lambda1s=None, n_lambda1: int = 13,
               decades: float = 3.0, keep_states: bool = True, point_callback=None,
               should_stop=None, **solve_kwargs) -> "LCurve":
        """Solve along decreasing lambda1 (lambda2 and the cross-gradient fixed) with
        warm starts.

        Default lambda1s: ``n_lambda1`` values from lambda1_max down
        ``decades`` decades (the first, lambda1_max itself, has the zero
        model and is skipped).  The corner is the maximum curvature of
        log10(misfit) against log10(group penalty), as for the L1–L2 path
        (``regparam.lcurve_corner_info``).
        """
        from .regparam import lcurve_corner_info

        top = self.lambda1_max()
        if lambda1s is None:
            lambda1s = top * 10.0 ** (-np.linspace(0, decades, n_lambda1 + 1)[1:])
        lambda1s = np.sort(np.asarray(lambda1s, dtype=float))[::-1]
        points, states = [], []
        state = None
        stopped = False
        for i, lam in enumerate(lambda1s):
            res = self.solve(lam, lambda2, state=state, should_stop=should_stop, **solve_kwargs)
            if res.stopped:
                stopped = True
                break
            state = res.state
            t = res.final_terms
            point = {"lambda1": float(lam), "misfit": t["misfit"],
                     "group_penalty": t["group"], "l2_penalty": t["l2"],
                     "cross_gradient": t["cross_gradient"],
                     "objective": self._objective(t, lam, lambda2, res.cross_gradient),
                     "n_iterations": res.n_iterations, "converged": res.converged,
                     "n_active": res.n_active}
            for d in self.datasets:
                point[f"misfit_{d.name}"] = t[f"misfit_{d.name}"]
            if res.gauss_newton:
                point["gn_iterations"] = len(res.gauss_newton)
            chi2 = self.chi2_of(res)
            if chi2 is not None:
                point.update({f"chi2_{k}": v for k, v in chi2.items() if k != "total"})
                point["chi2"] = chi2["total"]
            points.append(point)
            states.append(res.state if keep_states else None)
            if point_callback is not None:
                point_callback(i, len(lambda1s), points[-1])
        lc = LCurve(lambda2=float(lambda2), lambda1_max=top, points=points, states=states,
                    n_planned=len(lambda1s), stopped=stopped)
        ok = [p for p in points if p["group_penalty"] > 0 and p["misfit"] > 0]
        if len(ok) >= 4:
            try:
                corner = lcurve_corner_info([p["lambda1"] for p in ok], [p["misfit"] for p in ok],
                                            [p["group_penalty"] for p in ok])
                lc.corner = corner
                lc.lambda1_corner = corner["beta"]
            except ValueError:
                pass
        return lc

    def solve_at(self, lc: "LCurve", lambda1: float, lambda2: float, **solve_kwargs):
        """Solve at ``lambda1`` warm-started from the nearest larger point of ``lc``."""
        lams = np.array([p["lambda1"] for p in lc.points])
        k = int(np.searchsorted(-lams, -lambda1, side="right")) - 1
        state = lc.states[max(k, 0)] if lc.states and lc.states[max(k, 0)] is not None else None
        return self.solve(lambda1, lambda2, state=state, **solve_kwargs)


class JointGroupLassoProblem(GroupLassoProblem):
    """The magnetic–gravity problem of the paper (two linear datasets, two models).

    Args:
        magnetic_operator, gravity_operator: sensitivities K (N_f x M) and
            G (N_g x M) on the same M cells: dense arrays (float32 is kept),
            sparse matrices, or LinearOperators (then give the weights).
        magnetic_data, gravity_data: observed f and g (in the operators' units).
        sensitivity_weighting: True for w = ||column||^(-gamma/2), False for
            none (the operators are already scaled), or supply
            ``magnetic_weights`` / ``gravity_weights``.
        data_scaling: see :func:`data_scaling`; std_f / std_g for "std".
        mu: ADMM penalty; the zeta-update factorizations are made for it once
            and reused by every solve.  Default: the mean non-zero eigenvalue of
            X^T X and Y^T Y (M / min(N, M) with unit columns).
        solver: "auto", "cholesky" or "cg" (see :func:`make_solver`).
        mesh, active_cells: for the cross-gradient (see :class:`GroupLassoProblem`).
        bounds: [(lower, upper) of the susceptibility, (lower, upper) of the
            density], see :class:`GroupLassoProblem`.

    Memory: the operators are used as given (no weighted copies); the
    Cholesky solver adds one min(N, M)^2 float64 matrix per method, CG none.
    """

    def __init__(self, magnetic_operator, gravity_operator, magnetic_data, gravity_data,
                 mu: float | None = None, *, sensitivity_weighting: bool = True,
                 gamma: float = 2.0, magnetic_weights=None, gravity_weights=None,
                 data_scaling="max_ratio", std_f=None, std_g=None, solver: str = "auto",
                 factor_max_bytes: float = FACTOR_MAX_BYTES, cg_rtol: float = 1e-8,
                 cg_maxiter: int = 500, mesh=None, active_cells=None, bounds=None):
        K, G = magnetic_operator, gravity_operator
        if K.shape[1] != G.shape[1]:
            raise ValueError(f"The operators have {K.shape[1]} and {G.shape[1]} cells; "
                             "both models live on the same cells")
        f = np.asarray(magnetic_data, dtype=float).ravel()
        g = np.asarray(gravity_data, dtype=float).ravel()
        if f.size != K.shape[0] or g.size != G.shape[0]:
            raise ValueError("Data and operator rows do not match")
        super().__init__(
            [GroupLassoData("magnetic", f, 0, std_f, operator=K),
             GroupLassoData("gravity", g, 1, std_g, operator=G)],
            mu, model_names=["magnetic", "gravity"], sensitivity_weighting=sensitivity_weighting,
            gamma=gamma,
            model_weights=None if magnetic_weights is None and gravity_weights is None
            else [magnetic_weights, gravity_weights],
            data_scaling=data_scaling, solver=solver, factor_max_bytes=factor_max_bytes,
            cg_rtol=cg_rtol, cg_maxiter=cg_maxiter, mesh=mesh, active_cells=active_cells,
            bounds=bounds)

    # the paper's names
    f = property(lambda self: self.datasets[0].data)
    g = property(lambda self: self.datasets[1].data)
    std_f = property(lambda self: self.datasets[0].std)
    std_g = property(lambda self: self.datasets[1].std)
    s_f = property(lambda self: self.scales[0])
    s_g = property(lambda self: self.scales[1])
    c_f = property(lambda self: self.model_factors[0])
    c_g = property(lambda self: self.model_factors[1])
    bf = property(lambda self: self.b[0])
    bg = property(lambda self: self.b[1])
    X = property(lambda self: self.ops[0])
    Y = property(lambda self: self.ops[1])
    w_beta = property(lambda self: self.weights[0])
    w_rho = property(lambda self: self.weights[1])
    solver_X = property(lambda self: self.solvers[0])
    solver_Y = property(lambda self: self.solvers[1])
    Xt_f = property(lambda self: self.Zt_b[0])
    Yt_g = property(lambda self: self.Zt_b[1])

    def terms(self, s_beta, s_rho, pred_f=None, pred_g=None) -> dict:
        """Misfit 1/2 ||b - Z s||^2, group penalty and L2 penalty (unweighted by lambda)."""
        preds = None
        if pred_f is not None and pred_g is not None:   # scaled predictions, as before
            preds = [pred_f / self.s_f, pred_g / self.s_g]
        return self.terms_of([s_beta, s_rho], preds)

    def chi2(self, result: GroupLassoResult) -> dict | None:
        """chi^2 of each dataset (needs std_f and std_g)."""
        return self.chi2_of(result)

    def kkt_residual(self, s_beta, s_rho, lambda1: float, lambda2: float,
                     cross_gradient: float = 0.0) -> float:
        """Largest violation of the optimality conditions, relative to lambda1 (see
        :meth:`GroupLassoProblem.kkt_residuals`)."""
        return self.kkt_residuals([s_beta, s_rho], lambda1, lambda2, cross_gradient)


@dataclass
class LCurve:
    """Solutions along lambda1 at fixed lambda2 (see GroupLassoProblem.lcurve)."""

    lambda2: float
    lambda1_max: float
    points: list
    states: list = field(default_factory=list, repr=False)
    n_planned: int = 0
    stopped: bool = False
    corner: dict | None = None
    lambda1_corner: float | None = None

    def arrays(self) -> dict:
        return {k: np.array([p[k] for p in self.points]) for k in
                ("lambda1", "misfit", "group_penalty", "l2_penalty", "n_iterations")}


def plot_lcurve(lc: LCurve, ax=None, chosen: float | None = None):
    """log10(misfit) against log10(group penalty), with the corner marked."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    a = lc.arrays()
    ok = (a["misfit"] > 0) & (a["group_penalty"] > 0)
    x, y = np.log10(a["misfit"][ok]), np.log10(a["group_penalty"][ok])
    ax.plot(x, y, "o-", color="#2563eb", ms=4)
    for lam, xi, yi in zip(a["lambda1"][ok], x, y):
        ax.annotate(f"{lam:.2g}", (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    mark = chosen if chosen is not None else lc.lambda1_corner
    if mark is not None and ok.sum() >= 2:
        t = np.log(a["lambda1"][ok])[::-1]
        xm = np.interp(np.log(mark), t, x[::-1])
        ym = np.interp(np.log(mark), t, y[::-1])
        ax.plot([xm], [ym], "s", color="#dc2626", ms=8, label=f"λ1 = {mark:.3g}")
        ax.legend(fontsize=8)
    ax.set_xlabel("log10 data misfit ½‖b − Zζ‖²")
    ax.set_ylabel("log10 group penalty Σ‖ζ_k‖")
    ax.set_title(f"L-curve over λ1 (λ2 = {lc.lambda2:g})", fontsize=10)
    return ax


# ── Convenience wrapper ────────────────────────────────────────────────


def joint_group_lasso_admm(magnetic_operator, gravity_operator, magnetic_data, gravity_data,
                           lambda1: float, lambda2: float, mu: float | None = None,
                           max_iter: int = 2000,
                           tol_primal: float = 1e-4, tol_dual: float = 1e-4,
                           **kwargs) -> GroupLassoResult:
    """Solve the L2 + group-lasso joint problem once (see :class:`JointGroupLassoProblem`).

    Keyword arguments go to the problem (sensitivity_weighting, gamma,
    magnetic_weights, gravity_weights, data_scaling, std_f, std_g, solver,
    factor_max_bytes, cg_rtol, cg_maxiter, mesh, active_cells) or to
    :meth:`~GroupLassoProblem.solve` (tol_abs, state, history_every, callback,
    should_stop, coupling, cross_gradient).
    """
    solve_keys = {"tol_abs", "state", "history_every", "callback", "should_stop", "coupling",
                  "cross_gradient"}
    solve_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in solve_keys}
    problem = JointGroupLassoProblem(magnetic_operator, gravity_operator, magnetic_data,
                                     gravity_data, mu, **kwargs)
    return problem.solve(lambda1, lambda2, max_iter=max_iter, tol_primal=tol_primal,
                         tol_dual=tol_dual, **solve_kw)


# ── SimPEG ─────────────────────────────────────────────────────────────


def sensitivity_of(simulation, model=None, row_scale=None):
    """The sensitivity of a SimPEG simulation: (operator, column norms or None).

    An integral potential-field simulation that stores its sensitivities
    (``store_sensitivities="ram"``) gives ``simulation.G`` itself (float32, not
    copied).  Any other simulation is wrapped matrix-free with Jvec / Jtvec at
    ``model`` (default zeros; a linear problem's sensitivity does not depend on
    it), with its column norms from ``getJtJdiag`` (rows weighted by
    ``row_scale`` if given).
    """
    if getattr(simulation, "store_sensitivities", None) in ("ram", "disk"):
        return simulation.G, None
    if model is None:
        model = np.zeros(int(simulation.nC))
    n_data = int(simulation.survey.nD)
    op = LinearOperator((n_data, len(model)), dtype=float,
                        matvec=lambda v: simulation.Jvec(model, v),
                        rmatvec=lambda w: simulation.Jtvec(model, w))
    W = None if row_scale is None else sp.diags(np.broadcast_to(row_scale, (n_data,)))
    return op, np.sqrt(np.asarray(simulation.getJtJdiag(model, W=W), dtype=float))


def from_simulations(magnetic_simulation, gravity_simulation, magnetic_data, gravity_data,
                     mu: float | None = None, model=None, **kwargs) -> JointGroupLassoProblem:
    """The joint problem from two SimPEG simulations on the same active cells.

    The forward modelling stays SimPEG's; this only takes the sensitivities
    (see :func:`sensitivity_of`).  Keyword arguments as for
    :class:`JointGroupLassoProblem`.
    """
    gamma = kwargs.get("gamma", 2.0)
    std = kwargs.get("data_scaling") == "std"
    ops, traces = [], []
    for sim, key, sd in ((magnetic_simulation, "magnetic_weights", kwargs.get("std_f")),
                         (gravity_simulation, "gravity_weights", kwargs.get("std_g"))):
        op, norms = sensitivity_of(sim, model, 1.0 / np.asarray(sd, dtype=float) if std else None)
        if norms is not None:   # matrix-free: weights (and mu) from the column norms
            if kwargs.get(key) is None:
                kwargs[key] = sensitivity_weights(gamma=gamma, norms=norms)
            traces.append(float(np.sum((kwargs[key] * norms) ** 2)) / min(op.shape))
        ops.append(op)
    if mu is None and len(traces) == 2:
        mu = 0.5 * sum(traces)
    return JointGroupLassoProblem(ops[0], ops[1], magnetic_data, gravity_data, mu, **kwargs)
