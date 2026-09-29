"""Joint magnetic–gravity inversion with L2 + group-lasso regularization (Utsugi 2025).

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

ADMM (Boyd et al. 2011) with the split s = zeta and the scaled dual u:

    zeta <- (Z^T Z + mu I)^-1 (Z^T b + mu (s + u))     two independent systems
    s    <- per cell, with q = zeta - u:
            s_k = mu / (mu + lambda2) * max(1 - lambda1 / (mu ||q_k||), 0) * q_k
    u    <- u + s - zeta

until the primal residual ||s - zeta|| and the dual residual mu ||s - s_old||
are both within tolerance.  mu is the ADMM penalty: it changes how fast the
iteration gets there, not the minimizer.

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

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk
from scipy.sparse.linalg import LinearOperator, cg

# Largest N x N (or M x M) float64 matrix factorized per method before
# switching to conjugate gradients
FACTOR_MAX_BYTES = 4e9
# Column blocks of the sensitivity converted to float64 at a time
CHUNK_BYTES = 256e6


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


def data_scaling(f, g, mode="max_ratio", std_f=None, std_g=None):
    """Row scales (s_f, s_g) that bring the two datasets to comparable influence.

    mode:
        "max_ratio" (the paper): s_f = 1, s_g = C = max|f| / max|g|.
        "std": s = 1 / std (whitening), so the misfit is chi^2.
        "none": both 1.
        (s_f, s_g): explicit scalars or per-datum arrays.

    Returns (s_f, s_g, notes); notes say when a scale could not be computed.
    """
    notes = []
    if isinstance(mode, (tuple, list)):
        s_f, s_g = mode
    elif mode == "max_ratio":
        top_f, top_g = float(np.max(np.abs(f))), float(np.max(np.abs(g)))
        s_f = 1.0
        if top_f > 0 and top_g > 0:
            s_g = top_f / top_g
        else:
            s_g = 1.0
            notes.append("max|f| or max|g| is zero: the amplitude ratio C is undefined, "
                         "so the gravity data are not scaled (C = 1)")
    elif mode == "std":
        if std_f is None or std_g is None:
            raise ValueError("data_scaling='std' needs std_f and std_g")
        s_f = 1.0 / np.asarray(std_f, dtype=float)
        s_g = 1.0 / np.asarray(std_g, dtype=float)
    elif mode == "none":
        s_f = s_g = 1.0
    else:
        raise ValueError(f"Unknown data_scaling {mode!r} (expected 'max_ratio', 'std', "
                         "'none' or a pair of scales)")

    def check(s, n, name):
        s = float(s) if np.ndim(s) == 0 else np.asarray(s, dtype=float)
        if np.ndim(s) and np.shape(s) != (n,):
            raise ValueError(f"{name} has shape {np.shape(s)}, expected ({n},)")
        if not np.all(np.asarray(s) > 0) or not np.all(np.isfinite(s)):
            raise ValueError(f"{name} must be positive and finite")
        return s
    return check(s_f, len(f), "s_f"), check(s_g, len(g), "s_g"), notes


_data_scaling = data_scaling   # JointGroupLassoProblem's argument of that name shadows it


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


# ── Solvers for the zeta update: (X^T X + mu I) x = rhs ────────────────


class _DataSpaceCholesky:
    """Woodbury with a Cholesky factor of X X^T + mu I (fewer data than cells)."""

    name = "cholesky (data space)"

    def __init__(self, op: WeightedOperator, mu: float):
        self.op, self.mu = op, mu
        G = op.gram()
        G[np.diag_indices_from(G)] += mu
        self.factor = cho_factor(G, lower=False, overwrite_a=True, check_finite=False)

    def solve(self, rhs, x0=None):
        w = cho_solve(self.factor, self.op.matvec(rhs), check_finite=False)
        return (rhs - self.op.rmatvec(w)) / self.mu


class _ModelSpaceCholesky:
    """Cholesky factor of X^T X + mu I (fewer cells than data)."""

    name = "cholesky (model space)"

    def __init__(self, op: WeightedOperator, mu: float):
        H = op.normal()
        H[np.diag_indices_from(H)] += mu
        self.factor = cho_factor(H, lower=False, overwrite_a=True, check_finite=False)

    def solve(self, rhs, x0=None):
        return cho_solve(self.factor, rhs, check_finite=False)


class _ConjugateGradient:
    """Matrix-free CG on X^T X + mu I, warm-started from the previous zeta.

    Jacobi preconditioner when the column norms are known (with gamma = 2 the
    diagonal is constant and it changes nothing).
    """

    name = "conjugate gradients"

    def __init__(self, op: WeightedOperator, mu: float, rtol: float = 1e-8,
                 maxiter: int = 500, diag=None):
        m = op.shape[1]
        self.A = LinearOperator((m, m), dtype=float,
                                matvec=lambda v: op.rmatvec(op.matvec(v)) + mu * v)
        self.M = None if diag is None else LinearOperator(
            (m, m), dtype=float, matvec=lambda v: v / (diag + mu))
        self.rtol, self.maxiter = rtol, maxiter
        self.iterations = 0

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


def make_solver(op: WeightedOperator, mu: float, solver: str = "auto",
                factor_max_bytes: float = FACTOR_MAX_BYTES, cg_rtol: float = 1e-8,
                cg_maxiter: int = 500):
    """The zeta-update solver for one method.

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
        diag = None
        if op.explicit:   # diag(X^T X) = (w_j ||S K_j||)^2
            per_row = np.ndim(op.S) > 0
            diag = (op.w * column_norms(op.K, op.S if per_row else None)
                    * (1.0 if per_row else op.S)) ** 2
        return _ConjugateGradient(op, mu, rtol=cg_rtol, maxiter=cg_maxiter, diag=diag)
    raise ValueError(f"Unknown solver {solver!r} (expected 'auto', 'cholesky' or 'cg')")


# ── Group soft threshold ───────────────────────────────────────────────


def group_shrink(q_beta, q_rho, lambda1: float, lambda2: float, mu: float,
                 out_beta=None, out_rho=None, coupling: str = "group"):
    """Proximal step of lambda1 ||.||_group + lambda2/2 ||.||^2 at mu: the s update.

    s_k = mu / (mu + lambda2) * max(1 - lambda1 / (mu r_k), 0) * q_k with
    r_k = sqrt(q_beta_k^2 + q_rho_k^2): both components of a cell shrink
    together, and a cell with r_k <= lambda1 / mu (also r_k = 0) is emptied.
    O(M), vectorized.

    ``coupling="none"`` soft-thresholds beta and rho separately (ordinary
    elastic net on each model): a reference for comparisons, not the method.
    """
    damp = mu / (mu + lambda2)
    t = lambda1 / mu
    if coupling == "none":
        sb = np.sign(q_beta) * np.maximum(np.abs(q_beta) - t, 0.0) * damp
        sr = np.sign(q_rho) * np.maximum(np.abs(q_rho) - t, 0.0) * damp
        if out_beta is not None:
            out_beta[:], out_rho[:] = sb, sr
            return out_beta, out_rho
        return sb, sr
    if coupling != "group":
        raise ValueError(f"Unknown coupling {coupling!r} (expected 'group' or 'none')")
    r = np.hypot(q_beta, q_rho)
    # factor = damp * max(1 - t / r, 0), with r = 0 (and r <= t) giving 0
    factor = np.subtract(r, t, out=np.zeros_like(r), where=r > t)
    np.divide(factor, r, out=factor, where=r > t)
    factor *= damp
    out_beta = np.multiply(factor, q_beta, out=out_beta)
    out_rho = np.multiply(factor, q_rho, out=out_rho)
    return out_beta, out_rho


def group_norms(beta, rho) -> np.ndarray:
    return np.hypot(beta, rho)


# ── The problem ────────────────────────────────────────────────────────


@dataclass
class ADMMState:
    """Iterates of ADMM (scaled variables), for warm starts."""

    zeta_beta: np.ndarray
    zeta_rho: np.ndarray
    s_beta: np.ndarray
    s_rho: np.ndarray
    u_beta: np.ndarray
    u_rho: np.ndarray

    @classmethod
    def zeros(cls, m: int) -> "ADMMState":
        return cls(*(np.zeros(m) for _ in range(6)))

    def copy(self) -> "ADMMState":
        return ADMMState(*(a.copy() for a in (self.zeta_beta, self.zeta_rho, self.s_beta,
                                               self.s_rho, self.u_beta, self.u_rho)))


@dataclass
class GroupLassoResult:
    """Outcome of one ADMM solve.

    Models: ``*_scaled`` are the inversion variables (the s iterate, which has
    exact zeros); ``*_physical`` = weights * scaled / (scalar data scale), in
    the units of the given sensitivities (e.g. SI susceptibility and g/cc).
    Predicted data and residuals are in the units of the given data.
    Histories: ``misfit`` is 1/2 ||b - Z s||^2 (scaled data), ``group_penalty``
    sum_k ||s_k||, ``l2_penalty`` 1/2 ||s||^2 (without the lambdas);
    ``objective`` = misfit + lambda1 group + lambda2 l2.
    """

    beta_scaled: np.ndarray
    rho_scaled: np.ndarray
    beta_physical: np.ndarray
    rho_physical: np.ndarray
    predicted_magnetic: np.ndarray
    predicted_gravity: np.ndarray
    residual_magnetic: np.ndarray
    residual_gravity: np.ndarray
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

    @property
    def objective(self) -> float:
        return self.objective_history[-1] if self.objective_history else float("nan")

    @property
    def misfit(self) -> float:
        return self.misfit_history[-1] if self.misfit_history else float("nan")

    @property
    def group_penalty(self) -> float:
        return self.group_penalty_history[-1] if self.group_penalty_history else float("nan")


class JointGroupLassoProblem:
    """Scaled joint problem; solves it for any (lambda1, lambda2) at fixed mu.

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

    Memory: the operators are used as given (no weighted copies); the
    Cholesky solver adds one min(N, M)^2 float64 matrix per method, CG none.
    """

    def __init__(self, magnetic_operator, gravity_operator, magnetic_data, gravity_data,
                 mu: float | None = None, *, sensitivity_weighting: bool = True,
                 gamma: float = 2.0, magnetic_weights=None, gravity_weights=None,
                 data_scaling="max_ratio", std_f=None, std_g=None, solver: str = "auto",
                 factor_max_bytes: float = FACTOR_MAX_BYTES, cg_rtol: float = 1e-8,
                 cg_maxiter: int = 500):
        K, G = magnetic_operator, gravity_operator
        if K.shape[1] != G.shape[1]:
            raise ValueError(f"The operators have {K.shape[1]} and {G.shape[1]} cells; "
                             "both models live on the same cells")
        self.f = np.asarray(magnetic_data, dtype=float).ravel()
        self.g = np.asarray(gravity_data, dtype=float).ravel()
        if self.f.size != K.shape[0] or self.g.size != G.shape[0]:
            raise ValueError("Data and operator rows do not match")
        self.m = K.shape[1]
        self.gamma = gamma
        self.notes = []
        # standard deviations, when given, also report chi^2 (whatever the scaling)
        self.std_f = None if std_f is None else np.broadcast_to(
            np.asarray(std_f, dtype=float), self.f.shape)
        self.std_g = None if std_g is None else np.broadcast_to(
            np.asarray(std_g, dtype=float), self.g.shape)
        s_f, s_g, notes = _data_scaling(self.f, self.g, data_scaling, std_f, std_g)
        self.notes += notes
        for n in notes:
            warnings.warn(n)
        self.s_f, self.s_g = s_f, s_g
        self.bf = s_f * self.f      # scaled data b = [S_f f, S_g g]
        self.bg = s_g * self.g

        # A scalar data scale is carried by the model variable, as in the paper: the
        # operator keeps unit columns and scaled = S * physical / w, so the two
        # components of a group are in comparable (data) units.  A per-datum scale
        # (e.g. 1/std) cannot be, so it weights the operator's rows, and the column
        # weights are computed from the weighted rows.
        def split(s):
            return (s, 1.0) if np.ndim(s) else (1.0, s)   # (row scale, model factor)
        (r_f, self.c_f), (r_g, self.c_g) = split(s_f), split(s_g)

        def weights(given, op, row):
            """Column weights, and trace(X^T X) (None when unknown)."""
            explicit = isinstance(op, np.ndarray) or sp.issparse(op)
            norms = column_norms(op, row if np.ndim(row) else None) if explicit else None
            if given is not None:
                w = np.asarray(given, dtype=float)
            elif sensitivity_weighting:
                if norms is None:
                    raise ValueError("Give the weights of a matrix-free operator "
                                     "(magnetic_weights / gravity_weights)")
                w = sensitivity_weights(gamma=gamma, norms=norms)
            else:
                w = np.ones(op.shape[1])
            return w, None if norms is None else float(np.sum((w * norms) ** 2))
        self.w_beta, trace_x = weights(magnetic_weights, K, r_f)
        self.w_rho, trace_y = weights(gravity_weights, G, r_g)
        self.X = WeightedOperator(K, self.w_beta, r_f)
        self.Y = WeightedOperator(G, self.w_rho, r_g)
        if mu is None:
            # the mean non-zero eigenvalue of X^T X (M / min(N, M) for unit columns):
            # ADMM converges fastest for mu near the scale of the curvature
            if trace_x is None or trace_y is None:
                raise ValueError("mu has no default for a matrix-free operator; give it")
            mu = 0.5 * (trace_x / min(self.X.shape) + trace_y / min(self.Y.shape))
        if not mu > 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        self.mu = float(mu)
        t0 = time.time()
        self.solver_X = make_solver(self.X, self.mu, solver, factor_max_bytes, cg_rtol, cg_maxiter)
        self.solver_Y = make_solver(self.Y, self.mu, solver, factor_max_bytes, cg_rtol, cg_maxiter)
        self.setup_seconds = time.time() - t0
        self.Xt_f = self.X.rmatvec(self.bf)   # Z^T b, reused by every zeta update
        self.Yt_g = self.Y.rmatvec(self.bg)

    # -- quantities --------------------------------------------------------

    @property
    def solver_name(self) -> str:
        a, b = self.solver_X.name, self.solver_Y.name
        return a if a == b else f"{a} / {b}"

    def lambda1_max(self) -> float:
        """Smallest lambda1 whose solution is zero: max_k ||(Z^T b)_k||."""
        return float(np.max(np.hypot(self.Xt_f, self.Yt_g)))

    def terms(self, s_beta, s_rho, pred_f=None, pred_g=None) -> dict:
        """Misfit 1/2 ||b - Z s||^2, group penalty and L2 penalty (unweighted by lambda)."""
        if pred_f is None:
            pred_f = self.X.matvec(s_beta)
        if pred_g is None:
            pred_g = self.Y.matvec(s_rho)
        rf, rg = self.bf - pred_f, self.bg - pred_g
        return {"misfit": 0.5 * float(rf @ rf + rg @ rg),
                "misfit_magnetic": 0.5 * float(rf @ rf), "misfit_gravity": 0.5 * float(rg @ rg),
                "group": float(np.sum(np.hypot(s_beta, s_rho))),
                "l2": 0.5 * float(s_beta @ s_beta + s_rho @ s_rho)}

    def chi2(self, result: "GroupLassoResult") -> dict | None:
        """chi^2 of each dataset (needs std_f and std_g)."""
        if self.std_f is None or self.std_g is None:
            return None
        cf = float(np.sum((result.residual_magnetic / self.std_f) ** 2))
        cg = float(np.sum((result.residual_gravity / self.std_g) ** 2))
        return {"magnetic": cf, "gravity": cg, "total": cf + cg}

    def kkt_residual(self, s_beta, s_rho, lambda1: float, lambda2: float) -> float:
        """Largest violation of the optimality conditions, relative to lambda1.

        With c = Z^T (b - Z s): a cell with s_k != 0 needs
        c_k = lambda1 s_k / ||s_k|| + lambda2 s_k; an empty cell ||c_k|| <= lambda1.
        """
        cb = self.X.rmatvec(self.bf - self.X.matvec(s_beta))
        cr = self.Y.rmatvec(self.bg - self.Y.matvec(s_rho))
        r = np.hypot(s_beta, s_rho)
        on = r > 0
        scale = max(lambda1, 1e-300)
        viol = np.zeros(self.m)
        unit_b = np.divide(s_beta, r, out=np.zeros_like(r), where=on)
        unit_r = np.divide(s_rho, r, out=np.zeros_like(r), where=on)
        db = cb - lambda1 * unit_b - lambda2 * s_beta
        dr = cr - lambda1 * unit_r - lambda2 * s_rho
        viol[on] = np.hypot(db[on], dr[on]) / scale
        viol[~on] = np.maximum(np.hypot(cb[~on], cr[~on]) - lambda1, 0.0) / scale
        return float(viol.max()) if viol.size else 0.0

    # -- ADMM --------------------------------------------------------------

    def solve(self, lambda1: float, lambda2: float, *, max_iter: int = 2000,
              tol_primal: float = 1e-4, tol_dual: float = 1e-4, tol_abs: float | None = None,
              state: ADMMState | None = None, history_every: int = 1, callback=None,
              should_stop=None, coupling: str = "group") -> GroupLassoResult:
        """ADMM for one (lambda1, lambda2).

        Stops when r_primal = ||s - zeta|| <= sqrt(2M) tol_abs + tol_primal max(||zeta||, ||s||)
        and r_dual = mu ||s - s_old|| <= sqrt(2M) tol_abs + tol_dual ||mu u||
        (Boyd et al. 2011, section 3.3.1), or after ``max_iter`` iterations
        (``converged`` False).  Never on the objective alone.

        Args:
            tol_abs: absolute tolerance in scaled model units; default
                1e-9 max|Z^T b|.
            state: warm start (e.g. the previous lambda1 of a sweep).
            history_every: record objective terms every so many iterations
                (each record costs one product with X and one with Y).
            callback: ``callback(iteration, record)`` after each record.
            should_stop: ``() -> bool`` asked every iteration; True ends the
                solve with the current iterate (``stopped``).
        """
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be >= 0")
        t0 = time.time()
        mu, m = self.mu, self.m
        lam_max = self.lambda1_max()
        if tol_abs is None:
            tol_abs = 1e-9 * max(float(np.max(np.abs(np.concatenate([self.Xt_f, self.Yt_g])))), 1e-300)
        st = ADMMState.zeros(m) if state is None else state.copy()
        hist = {k: [] for k in ("objective", "misfit", "group", "l2", "primal", "dual")}
        sqrt_n = np.sqrt(2 * m)
        rhs = np.empty(m)
        s_old_b, s_old_r = np.empty(m), np.empty(m)
        q_b, q_r = np.empty(m), np.empty(m)
        converged = stopped = False
        it = 0

        if coupling == "group" and lambda1 >= lam_max:
            # the exact solution is zero (see lambda1_max); nothing to iterate
            st = ADMMState.zeros(m)
            converged = True
        while not converged and it < max_iter:
            if should_stop is not None and should_stop():
                stopped = True
                break
            it += 1
            # 1. zeta update: two independent systems
            np.add(st.s_beta, st.u_beta, out=rhs)
            rhs *= mu
            rhs += self.Xt_f
            st.zeta_beta = self.solver_X.solve(rhs, st.zeta_beta)
            np.add(st.s_rho, st.u_rho, out=rhs)
            rhs *= mu
            rhs += self.Yt_g
            st.zeta_rho = self.solver_Y.solve(rhs, st.zeta_rho)
            # 2. s update: group soft threshold of q = zeta - u
            s_old_b[:], s_old_r[:] = st.s_beta, st.s_rho
            np.subtract(st.zeta_beta, st.u_beta, out=q_b)
            np.subtract(st.zeta_rho, st.u_rho, out=q_r)
            group_shrink(q_b, q_r, lambda1, lambda2, mu, st.s_beta, st.s_rho, coupling)
            # 3. dual update
            st.u_beta += st.s_beta - st.zeta_beta
            st.u_rho += st.s_rho - st.zeta_rho
            # residuals
            r_pri = float(np.sqrt(np.sum((st.s_beta - st.zeta_beta) ** 2)
                                  + np.sum((st.s_rho - st.zeta_rho) ** 2)))
            r_dual = mu * float(np.sqrt(np.sum((st.s_beta - s_old_b) ** 2)
                                        + np.sum((st.s_rho - s_old_r) ** 2)))
            norm_x = max(np.sqrt(st.zeta_beta @ st.zeta_beta + st.zeta_rho @ st.zeta_rho),
                         np.sqrt(st.s_beta @ st.s_beta + st.s_rho @ st.s_rho))
            norm_y = mu * np.sqrt(st.u_beta @ st.u_beta + st.u_rho @ st.u_rho)
            eps_pri = sqrt_n * tol_abs + tol_primal * norm_x
            eps_dual = sqrt_n * tol_abs + tol_dual * norm_y
            hist["primal"].append(r_pri)
            hist["dual"].append(r_dual)
            converged = r_pri <= eps_pri and r_dual <= eps_dual
            if history_every and (it % history_every == 0 or converged or it == max_iter):
                t = self.terms(st.s_beta, st.s_rho)
                obj = t["misfit"] + lambda1 * t["group"] + lambda2 * t["l2"]
                for k, v in (("objective", obj), ("misfit", t["misfit"]), ("group", t["group"]),
                             ("l2", t["l2"])):
                    hist[k].append(v)
                if callback is not None:
                    callback(it, {"objective": obj, **t, "primal": r_pri, "dual": r_dual,
                                  "eps_primal": eps_pri, "eps_dual": eps_dual})

        return self._result(st, lambda1, lambda2, lam_max, hist, it, converged, stopped,
                            time.time() - t0)

    def to_physical(self, s_beta, s_rho):
        """Models in the operators' units: w * scaled / (scalar data scale)."""
        return self.w_beta * s_beta / self.c_f, self.w_rho * s_rho / self.c_g

    def _result(self, st, lambda1, lambda2, lam_max, hist, it, converged, stopped, seconds):
        beta_phys, rho_phys = self.to_physical(st.s_beta, st.s_rho)
        pred_f = self.X.matvec(st.s_beta) / self.s_f     # = K beta_phys
        pred_g = self.Y.matvec(st.s_rho) / self.s_g
        t = self.terms(st.s_beta, st.s_rho, self.s_f * pred_f, self.s_g * pred_g)
        if not hist["objective"] or it == 0:
            hist["objective"].append(t["misfit"] + lambda1 * t["group"] + lambda2 * t["l2"])
            hist["misfit"].append(t["misfit"])
            hist["group"].append(t["group"])
            hist["l2"].append(t["l2"])
        return GroupLassoResult(
            beta_scaled=st.s_beta.copy(), rho_scaled=st.s_rho.copy(),
            beta_physical=beta_phys, rho_physical=rho_phys,
            predicted_magnetic=pred_f, predicted_gravity=pred_g,
            residual_magnetic=self.f - pred_f, residual_gravity=self.g - pred_g,
            objective_history=hist["objective"], misfit_history=hist["misfit"],
            group_penalty_history=hist["group"], l2_penalty_history=hist["l2"],
            primal_residual_history=hist["primal"], dual_residual_history=hist["dual"],
            n_iterations=it, converged=bool(converged), lambda1=float(lambda1),
            lambda2=float(lambda2), mu=self.mu, lambda1_max=lam_max,
            data_scale=(self.s_f, self.s_g), solver=self.solver_name, stopped=stopped,
            n_active=int(np.count_nonzero(np.hypot(st.s_beta, st.s_rho))),
            seconds=seconds, notes=list(self.notes), state=st)

    # -- L-curve over lambda1 ----------------------------------------------

    def lcurve(self, lambda2: float, lambda1s=None, n_lambda1: int = 13,
               decades: float = 3.0, keep_states: bool = True, point_callback=None,
               should_stop=None, **solve_kwargs) -> "LCurve":
        """Solve along decreasing lambda1 (lambda2 fixed) with warm starts.

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
            points.append({"lambda1": float(lam), "misfit": res.misfit,
                           "group_penalty": res.group_penalty,
                           "l2_penalty": res.l2_penalty_history[-1],
                           "objective": res.objective, "n_iterations": res.n_iterations,
                           "converged": res.converged, "n_active": res.n_active,
                           "misfit_magnetic": 0.5 * float(np.sum((self.s_f * res.residual_magnetic) ** 2)),
                           "misfit_gravity": 0.5 * float(np.sum((self.s_g * res.residual_gravity) ** 2))})
            chi2 = self.chi2(res)
            if chi2 is not None:
                points[-1].update(chi2_magnetic=chi2["magnetic"], chi2_gravity=chi2["gravity"],
                                  chi2=chi2["total"])
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


@dataclass
class LCurve:
    """Solutions along lambda1 at fixed lambda2 (see JointGroupLassoProblem.lcurve)."""

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
    factor_max_bytes, cg_rtol, cg_maxiter) or to :meth:`~JointGroupLassoProblem.solve`
    (tol_abs, state, history_every, callback, should_stop, coupling).
    """
    solve_keys = {"tol_abs", "state", "history_every", "callback", "should_stop", "coupling"}
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
