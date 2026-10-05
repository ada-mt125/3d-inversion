"""Linear-Gaussian Bayesian inversion of potential-field data, sampled by randomized
maximum likelihood (RML).

Gravity and magnetic (induced) data are linear in the model, ``d = G m + e`` with Gaussian
errors ``e ~ N(0, C_d)``, ``C_d = diag(sigma^2)``.  The prior is the smooth L2
regularization's Gaussian, ``m ~ N(m_ref, (beta P)^-1)`` with ``P = sum_i a_i B_i^T B_i``
(the smallness and smoothness terms, with their depth or sensitivity weights): the
posterior mean is then the smooth L2 model at that beta.  beta is chosen, as the
deterministic runs do, so that the mean fits the data to chi^2 = N (empirical Bayes).

Everything goes through data space, with ``Q = P^-1 G^T`` (one sparse factorization of P,
N solves) and the eigen-decomposition of the whitened ``Wd G Q Wd = V L V^T`` (N x N):

    mean      m = m_ref + Q Wd V diag(1 / (l + beta)) V^T Wd (d - G m_ref)
    chi^2(b)  = sum_i (b c_i / (l_i + b))^2,   c = V^T Wd (d - G m_ref)

so beta for chi^2 = N is found exactly by bisection, or beta and an error scale s (sigma ->
s sigma) by the largest evidence, ``log p(d | beta, s^2) = -1/2 sum_i [log(l_i / beta +
s^2) + c_i^2 / (l_i / beta + s^2)]`` (the mean then depends on beta s^2), and each RML sample (the data plus
an error drawn from C_d, the reference plus a field drawn from the prior, the mean of that
problem) costs two products with Q and G.  The samples' spread is the posterior's.
"""

from __future__ import annotations

import math
import time

import numpy as np


def prior_terms(reg, m):
    """The regularization's quadratic terms at ``m``: [(multiplier, B)] with each term's
    value ``||B (m - m_ref)||^2`` (SimPEG's W f_m'), for a WeightedLeastSquares."""
    terms = []
    for mult, f in zip(reg.multipliers, reg.objfcts):
        if not mult:
            continue
        B = f.W @ f.f_m_deriv(m)
        terms.append((float(mult), B.tocsr()))
    return terms


def precision(terms):
    P = None
    for mult, B in terms:
        T = mult * (B.T @ B)
        P = T if P is None else P + T
    return P.tocsc()


class LinearGaussian:
    """The posterior of ``d = G m + e`` under the prior given by ``terms`` (see module)."""

    def __init__(self, G, d, sigma, terms, m_ref=None, progress=None, chunk=256):
        from scipy.sparse.linalg import splu
        t0 = time.time()
        self.G = np.asarray(G, dtype=np.float32)
        self.d = np.asarray(d, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.terms = terms
        N, M = self.G.shape
        self.m_ref = np.zeros(M) if m_ref is None else np.asarray(m_ref, dtype=float)
        self.lu = splu(precision(terms), permc_spec="COLAMD")
        self.timing = {"factorize_s": time.time() - t0}
        # Q = P^-1 G^T, a block of columns at a time
        t1 = time.time()
        Q = np.empty((M, N), dtype=np.float32)
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            Q[:, a:b] = self.lu.solve(np.asarray(self.G[a:b].T, dtype=float))
            if progress:
                progress(a=b, n=N)
        self.Q = Q
        self.timing["solve_s"] = time.time() - t1
        t2 = time.time()
        wd = 1.0 / self.sigma
        K = (self.G @ Q).astype(float)
        K = 0.5 * (K + K.T) * wd[:, None] * wd[None, :]
        lam, V = np.linalg.eigh(K)
        self.lam = np.clip(lam, 0.0, None)
        self.V = V
        self.wd = wd
        self.timing["eig_s"] = time.time() - t2

    # ── the mean for a beta, and beta for chi^2 = target ──
    def _coefficients(self, d, m_ref):
        return self.V.T @ (self.wd * (d - self.G @ m_ref))

    def chi2(self, beta, c):
        return float(np.sum((beta * c / (self.lam + beta)) ** 2))

    def beta_for(self, target: float, c=None) -> float:
        """beta with chi^2 of the mean = ``target`` (bisection in log beta)."""
        c = self._coefficients(self.d, self.m_ref) if c is None else c
        lo, hi = 1e-12 * max(self.lam.max(), 1e-30), 1e12 * max(self.lam.max(), 1e-30)
        if self.chi2(hi, c) < target:      # even the reference fits: the largest beta
            return hi
        if self.chi2(lo, c) > target:      # the data cannot be fitted to the target
            return lo
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if self.chi2(mid, c) > target:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1 + 1e-6:
                break
        return math.sqrt(lo * hi)

    def log_evidence(self, beta, c=None) -> float:
        """log p(d | beta) up to a constant: d - G m_ref ~ N(0, G (beta P)^-1 G^T + C_d), so
        -1/2 sum_i [log(l_i / beta + 1) + c_i^2 / (l_i / beta + 1)]."""
        c = self._coefficients(self.d, self.m_ref) if c is None else c
        t = self.lam / beta + 1.0
        return float(-0.5 * np.sum(np.log(t) + c ** 2 / t))

    def beta_evidence(self, c=None) -> float:
        """beta with the largest evidence (type-II maximum likelihood), on a log grid refined
        around its best point."""
        c = self._coefficients(self.d, self.m_ref) if c is None else c
        top = max(self.lam.max(), 1e-30)
        grid = top * np.logspace(-14, 4, 181)
        ev = np.array([self.log_evidence(b, c) for b in grid])
        k = int(np.argmax(ev))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        for _ in range(60):                       # golden section in log beta
            a, b = math.log(lo), math.log(hi)
            m1, m2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
            if self.log_evidence(math.exp(m1), c) > self.log_evidence(math.exp(m2), c):
                hi = math.exp(m2)
            else:
                lo = math.exp(m1)
        return math.sqrt(lo * hi)

    def log_evidence2(self, beta, noise2, c=None) -> float:
        """log p(d | beta, s^2) with the errors scaled to s sigma: -1/2 sum_i
        [log(l_i / beta + s^2) + c_i^2 / (l_i / beta + s^2)] (up to the same constant)."""
        c = self._coefficients(self.d, self.m_ref) if c is None else c
        t = self.lam / beta + noise2
        return float(-0.5 * np.sum(np.log(t) + c ** 2 / t))

    def evidence_beta_noise(self, c=None) -> tuple:
        """(beta, s^2) with the largest evidence: the prior's scale and the errors' scale
        (sigma -> s sigma) estimated together from the data (type-II maximum likelihood)."""
        from scipy.optimize import minimize
        c = self._coefficients(self.d, self.m_ref) if c is None else c
        top = max(self.lam.max(), 1e-30)
        best = None
        for lb in np.log(top) + np.linspace(-32, 9, 42):            # a coarse grid first
            for ls in np.linspace(-9, 5, 29):
                v = self.log_evidence2(math.exp(lb), math.exp(ls), c)
                if best is None or v > best[0]:
                    best = (v, lb, ls)
        res = minimize(lambda x: -self.log_evidence2(math.exp(x[0]), math.exp(x[1]), c),
                       x0=[best[1], best[2]], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
        lb, ls = res.x if res.fun <= -best[0] else (best[1], best[2])
        return math.exp(lb), math.exp(ls)

    def mean(self, beta, d=None, m_ref=None):
        d = self.d if d is None else d
        m_ref = self.m_ref if m_ref is None else m_ref
        c = self._coefficients(d, m_ref)
        return m_ref + self.Q @ (self.wd * (self.V @ (c / (self.lam + beta))))

    # ── RML samples ──
    def prior_sample(self, beta, rng):
        """A draw from N(0, (beta P)^-1): P^-1 sum_i sqrt(a_i) B_i^T z_i / sqrt(beta)."""
        rhs = None
        for mult, B in self.terms:
            v = math.sqrt(mult) * (B.T @ rng.standard_normal(B.shape[0]))
            rhs = v if rhs is None else rhs + v
        return self.lu.solve(rhs) / math.sqrt(beta)

    def samples(self, beta, n, seed=0, progress=None, noise2=1.0):
        """``n`` RML samples of the posterior (errors s sigma with s^2 = ``noise2``), and the
        prior draws they started from."""
        rng = np.random.default_rng(seed)
        M = self.G.shape[1]
        out = np.empty((n, M), dtype=np.float32)
        prior = np.empty((n, M), dtype=np.float32)
        s = math.sqrt(noise2)
        for k in range(n):
            x = self.prior_sample(beta, rng)
            e = s * self.sigma * rng.standard_normal(len(self.d))
            out[k] = self.mean(beta * noise2, d=self.d + e, m_ref=self.m_ref + x)
            prior[k] = x
            if progress:
                progress(k=k + 1, n=n)
        return out, prior


def body_threshold(mean, core=None, share=0.25, weights=None) -> float:
    """A body: above ``share`` of the 98th percentile of the mean's positive values (in the
    ``core`` cells when given; ``weights``, e.g. cell volumes, weigh the percentile)."""
    v = np.asarray(mean, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    if core is not None:
        keep = np.asarray(core, bool)
        v, w = v[keep], w[keep]
    keep = np.isfinite(v) & (v > 0)
    v, w = v[keep], w[keep]
    if not v.size:
        return float("inf")
    order = np.argsort(v)
    cw = np.cumsum(w[order]) / w.sum()
    return share * float(v[order][min(np.searchsorted(cw, 0.98), len(v) - 1)])
