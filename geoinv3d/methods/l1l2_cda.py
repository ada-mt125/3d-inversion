"""L1–L2 (elastic net) inversion by coordinate descent, after Utsugi (2019).

Utsugi (2019, Earth Planets Space 71:73) inverts potential-field data
``f = K b*`` (b* = magnetization, A/m) by minimizing

    L(b; lam, a) = 1/2 ||f - X b||^2 + lam [ (1 - a)/2 ||b||^2 + a ||b||_1 ]

with the weighted variables ``b_j = s_j b*_j`` and columns ``x_j = k_j / s_j``.
Two depth weightings are compared in the paper:

    wS1:  s_j = ||k_j||^(1/2)   (Li & Oldenburg 2000)
    wS2:  s_j = ||k_j||         (unit columns, as in LASSO; recommended)

The problem is solved by the coordinate descent algorithm (CDA; Friedman et al.
2007, 2010) with the soft-threshold update of the paper's Eq. (26),

    b_j <- S(x_j^T r_-j, lam a) / (x_j^T x_j + lam (1 - a)),

optionally clipped to bounds (Eqs. 19-20), sweeping until
``||b^(k+1) - b^(k)|| / ||b^(k)|| < 1e-5``.  Solutions are computed along a
decreasing sequence of lam from ``lam_max = max_j |x_j^T f| / a`` (where b = 0)
with warm starts, at a step of 0.1 in log10(lam).  The paper picks lam at the
maximum curvature of the log-log L-curve ``||f - X b||`` vs ``P(b; a)``,
interpolated by cubic splines, with the mixing ratio ``a`` fixed a priori
(about 0.96 for wS1 and 0.90 for wS2 in its synthetic tests).

Two further criteria are offered on the same path: the discrepancy principle
(chi^2 = N) and GCV with the elastic-net degrees of freedom
``df = tr[X_A (X_A^T X_A + lam (1 - a) I)^-1 X_A^T]`` over the free active set
(Zou & Hastie 2005).

Data are not whitened in the paper (uniform noise).  ``data_weights`` may be
given for non-uniform errors; the worker passes ``mean(std) / std``, which is
the identity for uniform errors, so ``a`` keeps the paper's meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # numba makes the coordinate sweeps ~100x faster; the algorithm is the same
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba
    njit = None


WEIGHTINGS = {"S1": 0.5, "S2": 1.0}  # s_j = ||k_j|| ** exponent


def _sweep_loop(X, xtx, beta, r, lam1, lam2, lo, hi, active_only):
    """One coordinate sweep in place; returns ||delta beta||^2."""
    n, m = X.shape
    change2 = 0.0
    for j in range(m):
        bj = beta[j]
        if active_only and bj == 0.0:
            continue
        rho = xtx[j] * bj
        for i in range(n):
            rho += X[i, j] * r[i]
        if rho > lam1:
            new = (rho - lam1) / (xtx[j] + lam2)
        elif rho < -lam1:
            new = (rho + lam1) / (xtx[j] + lam2)
        else:
            new = 0.0
        if new < lo[j]:
            new = lo[j]
        elif new > hi[j]:
            new = hi[j]
        delta = new - bj
        if delta != 0.0:
            for i in range(n):
                r[i] -= X[i, j] * delta
            beta[j] = new
            change2 += delta * delta
    return change2


def _sweep_numpy(X, xtx, beta, r, lam1, lam2, lo, hi, active_only):
    """The same sweep with numpy inner products (fallback without numba)."""
    change2 = 0.0
    cols = np.flatnonzero(beta) if active_only else range(X.shape[1])
    for j in cols:
        bj = beta[j]
        rho = xtx[j] * bj + X[:, j] @ r
        new = np.sign(rho) * max(abs(rho) - lam1, 0.0) / (xtx[j] + lam2)
        new = min(max(new, lo[j]), hi[j])
        delta = new - bj
        if delta != 0.0:
            r -= X[:, j] * delta
            beta[j] = new
            change2 += delta * delta
    return change2


_sweep = njit(cache=True)(_sweep_loop) if njit is not None else _sweep_numpy


def _polish(X, f, beta, lam1, lam2, lo, hi):
    """Exact solution for the current active set and signs, or None.

    With the non-zero set A (free part F, the rest held at bounds) and the
    signs of b fixed, the optimality conditions are linear:
    ``(X_F^T X_F + lam2 I) b_F = X_F^T (f - X_B b_B) - lam1 sign(b_F)``.
    The solution is accepted only if it keeps the signs and the bounds; the
    caller's next full CDA sweep then confirms (or corrects) optimality.
    """
    active = beta != 0.0
    free = active & (beta > lo) & (beta < hi)
    if not free.any():
        return None
    held = active & ~free
    XF = X[:, free]
    rhs_data = f - X[:, held] @ beta[held] if held.any() else f
    sign = np.sign(beta[free])
    H = XF.T @ XF
    H[np.diag_indices_from(H)] += lam2
    try:
        new = np.linalg.solve(H, XF.T @ rhs_data - lam1 * sign)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(new)) or np.any(np.sign(new) != sign) \
            or np.any(new < lo[free]) or np.any(new > hi[free]):
        return None
    out = beta.copy()
    out[free] = new
    return out


def coordinate_descent(X, f, lam: float, alpha: float, beta0=None, lower=None, upper=None,
                       xtx=None, tol: float = 1e-5, max_sweeps: int = 100_000,
                       polish_every: int = 20):
    """Minimize ``1/2 ||f - X b||^2 + lam [(1-a)/2 ||b||^2 + a ||b||_1]`` by CDA.

    Full sweeps alternate with sweeps over the non-zero coefficients (the
    active set) until a full sweep changes the solution by less than ``tol``
    relative to its norm (the paper uses 1e-5).

    Columns of potential-field kernels are strongly correlated, so CDA can
    need many thousands of sweeps near a = 1.  Every ``polish_every`` active
    sweeps the active-set optimality conditions are solved directly (see
    :func:`_polish`); a candidate is kept only if it preserves signs and
    bounds and, as always, only a full sweep that changes nothing ends the
    iteration, so the result is the same minimizer.  ``polish_every=0``
    gives plain CDA.

    Args:
        X: (N, M) matrix; Fortran order is fastest.
        f: (N,) data.
        beta0: Warm start (default 0).
        lower, upper: Optional bounds on b, scalars or (M,) arrays.

    Returns:
        (b, n_sweeps)
    """
    X = np.asfortranarray(X, dtype=float)
    f = np.asarray(f, dtype=float)
    m = X.shape[1]
    beta = np.zeros(m) if beta0 is None else np.array(beta0, dtype=float)
    lo = np.broadcast_to(-np.inf if lower is None else lower, (m,)).astype(float)
    hi = np.broadcast_to(np.inf if upper is None else upper, (m,)).astype(float)
    beta = np.clip(beta, lo, hi)
    if xtx is None:
        xtx = np.einsum("ij,ij->j", X, X)
    r = f - X @ beta
    lam1, lam2 = lam * alpha, lam * (1.0 - alpha)

    def converged(change2):
        return change2 <= tol**2 * max(beta @ beta, 1e-300)

    sweeps = 0
    while sweeps < max_sweeps:
        change2 = _sweep(X, xtx, beta, r, lam1, lam2, lo, hi, False)
        sweeps += 1
        if converged(change2):
            break
        inner = 0
        while sweeps < max_sweeps:  # converge on the active set first
            change2 = _sweep(X, xtx, beta, r, lam1, lam2, lo, hi, True)
            sweeps += 1
            inner += 1
            if converged(change2):
                break
            if polish_every and inner % polish_every == 0:
                polished = _polish(X, f, beta, lam1, lam2, lo, hi)
                if polished is not None:
                    beta[:] = polished
                    r[:] = f - X @ beta
                    break  # let a full sweep check optimality
    return beta, sweeps


def penalty(beta, alpha: float) -> float:
    """P(b; a) = (1-a)/2 ||b||^2 + a ||b||_1 (the paper's Eq. 21 without lam)."""
    return float(0.5 * (1.0 - alpha) * beta @ beta + alpha * np.abs(beta).sum())


def elastic_net_dof(X, beta, lam: float, alpha: float, free=None) -> float:
    """Degrees of freedom tr[X_A (X_A^T X_A + lam(1-a) I)^-1 X_A^T].

    A is the set of non-zero coefficients not held at a bound (``free``).
    Computed from the eigenvalues of the N x N matrix X_A X_A^T.
    """
    active = beta != 0.0
    if free is not None:
        active &= free
    if not active.any():
        return 0.0
    XA = X[:, active]
    e = np.clip(np.linalg.eigvalsh(XA @ XA.T), 0.0, None)
    c = lam * (1.0 - alpha)
    if c == 0.0:
        return float(np.sum(e > 1e-10 * e.max()))
    return float(np.sum(e / (e + c)))


@dataclass
class ElasticNetPath:
    """Solutions of the elastic net along a decreasing sequence of lam."""

    alpha: float
    lambdas: np.ndarray
    betas: np.ndarray            # (n_lambda, M) weighted variables b
    residual_norm: np.ndarray    # ||f - X b|| (weighted data)
    penalty: np.ndarray          # P(b; a)
    sweeps: np.ndarray
    dof: np.ndarray = field(default_factory=lambda: np.array([]))


def lambda_max(X, f, alpha: float) -> float:
    """Smallest lam with b = 0: max_j |x_j^T f| / a (Friedman et al. 2010)."""
    return float(np.max(np.abs(X.T @ f)) / max(alpha, 1e-3))


def elastic_net_path(X, f, alpha: float, lambdas=None, n_decades: float = 4.0,
                     step: float = 0.1, lower=None, upper=None, tol: float = 1e-5,
                     with_dof: bool = False) -> ElasticNetPath:
    """CDA with warm starts along ``lambdas`` (default: lam_max down n_decades)."""
    X = np.asfortranarray(X, dtype=float)
    if lambdas is None:
        top = lambda_max(X, f, alpha)
        n = int(round(n_decades / step)) + 1
        lambdas = top * 10.0 ** (-step * np.arange(n))
    lambdas = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    xtx = np.einsum("ij,ij->j", X, X)
    free = None
    beta = None
    betas, res, pen, sweeps, dof = [], [], [], [], []
    for lam in lambdas:
        beta, k = coordinate_descent(X, f, lam, alpha, beta0=beta, lower=lower,
                                     upper=upper, xtx=xtx, tol=tol)
        betas.append(beta.copy())
        res.append(float(np.linalg.norm(f - X @ beta)))
        pen.append(penalty(beta, alpha))
        sweeps.append(k)
        if with_dof:
            free = _free_mask(beta, lower, upper)
            dof.append(elastic_net_dof(X, beta, lam, alpha, free))
    return ElasticNetPath(alpha=alpha, lambdas=lambdas, betas=np.array(betas),
                          residual_norm=np.array(res), penalty=np.array(pen),
                          sweeps=np.array(sweeps), dof=np.array(dof))


def _free_mask(beta, lower, upper):
    free = np.ones(beta.shape, dtype=bool)
    if lower is not None:
        free &= beta > np.broadcast_to(lower, beta.shape)
    if upper is not None:
        free &= beta < np.broadcast_to(upper, beta.shape)
    return free


def column_scaling(K, weighting: str = "S2") -> np.ndarray:
    """s_j for the weighted variables b_j = s_j b*_j (see module docstring)."""
    try:
        exponent = WEIGHTINGS[weighting]
    except KeyError:
        raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}, "
                         f"got {weighting!r}") from None
    norms = np.linalg.norm(K, axis=0)
    norms = np.maximum(norms, 1e-12 * norms.max())
    return norms**exponent


@dataclass
class L1L2Result:
    """Outcome of :func:`invert_l1l2`; models are in the caller's units."""

    alpha: float
    weighting: str
    criterion: str
    lambda_opt: float
    model: np.ndarray                 # recovered model (e.g. susceptibility)
    predicted: np.ndarray             # predicted data (unweighted)
    path: ElasticNetPath
    lambda_lcurve: float
    lambda_discrepancy: float | None = None
    lambda_gcv: float | None = None
    gcv: np.ndarray | None = None
    chi2: np.ndarray | None = None    # whitened misfit along the path
    warnings: list = field(default_factory=list)

    def path_models(self, scale: np.ndarray, unit: float) -> np.ndarray:
        """All path models in the caller's units."""
        return self.path.betas / (scale * unit)


def invert_l1l2(G, d, alpha: float, weighting: str = "S2", model_unit: float = 1.0,
                std=None, criterion: str = "lcurve", lower=None, upper=None,
                n_decades: float = 4.0, step: float = 0.1, tol: float = 1e-5
                ) -> tuple[L1L2Result, np.ndarray]:
    """Utsugi (2019) L1–L2 inversion with the regularization parameter chosen on the path.

    Args:
        G: (N, M) sensitivity matrix, data per unit model (e.g. nT per SI).
        d: (N,) observed data.
        alpha: Mixing ratio a in [0, 1], fixed a priori.
        weighting: "S2" (s_j = ||k_j||, recommended) or "S1" (||k_j||^(1/2)).
        model_unit: Physical unit of b* per model unit, so that
            ``K = G / model_unit`` and ``b* = model_unit * m``; for
            susceptibility and the paper's magnetization, the inducing field
            ``B0 / mu0`` in A/m.  Affects only wS1 (wS2 is unit-free).
        std: Optional data standard deviations.  Rows are weighted by
            ``mean(std) / std`` (identity for uniform errors, as in the
            paper); needed for the discrepancy criterion.
        criterion: "lcurve" (the paper), "discrepancy" (chi^2 = N, needs
            std) or "gcv".
        lower, upper: Optional bounds on the model (caller's units).
        n_decades, step: lam sequence from lam_max, in log10 units.

    Returns:
        (result, scale) where ``scale`` holds s_j; the weighted variable is
        ``b_j = s_j * model_unit * m_j``.
    """
    from .regparam import gcv_minimum, gcv_score, lcurve_corner

    if criterion not in ("lcurve", "discrepancy", "gcv"):
        raise ValueError(f"Unknown criterion '{criterion}'")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    n_data = len(d)
    if std is not None:
        std = np.broadcast_to(np.asarray(std, dtype=float), d.shape)
        w = std.mean() / std
    else:
        w = np.ones(n_data)
    if criterion == "discrepancy" and std is None:
        raise ValueError("The discrepancy criterion needs data standard deviations")

    K = (G / model_unit) * w[:, None]
    f = d * w
    s = column_scaling(K, weighting)
    X = np.asfortranarray(K / s)
    to_b = s * model_unit  # b = to_b * m
    lo = None if lower is None else np.asarray(lower, dtype=float) * to_b
    hi = None if upper is None else np.asarray(upper, dtype=float) * to_b

    path = elastic_net_path(X, f, alpha, n_decades=n_decades, step=step, lower=lo,
                            upper=hi, tol=tol, with_dof=(criterion == "gcv"))
    warnings = []
    # Near lam_max, b ~ 0 and log P -> -inf; such points (P = 0, or a lone
    # coefficient left by rounding) make the spline ring and fake a corner.
    ok = path.penalty > 1e-6 * path.penalty.max()
    lam_lc = lcurve_corner(path.lambdas[ok], path.residual_norm[ok], path.penalty[ok])
    inner = np.sort(path.lambdas[ok])[[1, -2]]
    if not inner[0] * 1.001 < lam_lc < inner[1] / 1.001:
        warnings.append("L-curve corner is at the edge of the lambda range; "
                        "extend n_decades")

    chi2 = None
    lam_disc = None
    if std is not None:
        sigma_w = std.mean()  # whitened residual = weighted residual / mean(std)
        chi2 = (path.residual_norm / sigma_w) ** 2
        lam_disc = _discrepancy_lambda(path.lambdas, chi2, n_data)
        if lam_disc is None:
            warnings.append("chi^2 = N is not reached on the lambda path")

    gcv = lam_gcv = None
    if criterion == "gcv":
        gcv = np.array([gcv_score(r**2, df, n_data)
                        for r, df in zip(path.residual_norm, path.dof)])
        lam_gcv = gcv_minimum(path.lambdas, gcv)

    lam_opt = {"lcurve": lam_lc, "discrepancy": lam_disc, "gcv": lam_gcv}[criterion]
    if lam_opt is None:
        raise ValueError("chi^2 = N is not reached on the lambda path; extend n_decades")
    # Solve at the chosen lam, warm-started from the nearest larger path lam
    k = int(np.searchsorted(-path.lambdas, -lam_opt, side="right")) - 1
    beta0 = path.betas[max(k, 0)]
    beta, _ = coordinate_descent(X, f, lam_opt, alpha, beta0=beta0, lower=lo, upper=hi,
                                 tol=tol)
    model = beta / to_b
    result = L1L2Result(alpha=alpha, weighting=weighting, criterion=criterion,
                        lambda_opt=float(lam_opt), model=model, predicted=G @ model,
                        path=path, lambda_lcurve=lam_lc, lambda_discrepancy=lam_disc,
                        lambda_gcv=lam_gcv, gcv=gcv, chi2=chi2, warnings=warnings)
    return result, s


def _discrepancy_lambda(lambdas, chi2, n_data):
    """lam where chi^2 crosses N, interpolated in log(lam) and log(chi^2)."""
    lambdas, chi2 = np.asarray(lambdas), np.asarray(chi2)  # lam decreasing
    above = chi2 >= n_data
    for i in range(len(lambdas) - 1):
        if above[i] and not above[i + 1]:
            t = (np.log(n_data) - np.log(chi2[i])) / (np.log(chi2[i + 1]) - np.log(chi2[i]))
            return float(np.exp(np.log(lambdas[i]) + t * (np.log(lambdas[i + 1])
                                                          - np.log(lambdas[i]))))
    return None
