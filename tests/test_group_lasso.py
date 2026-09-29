"""Joint magnetic–gravity inversion with L2 + group lasso (Utsugi 2025), solved by ADMM."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

from geoinv3d.methods import group_lasso as gl

NX, NY, NZ, H = 16, 16, 8, 50.0


@pytest.fixture(scope="module")
def kernels():
    """Magnetic (TMI, nT per SI) and gravity (gz, mGal per g/cc) sensitivities of
    16 x 16 x 8 cells of 50 m under 12 x 12 stations."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod

    mesh = Mesh3D.uniform(NX, NY, NZ, H, H, H, origin=(0.0, 0.0, -NZ * H))
    xy = np.linspace(25, NX * H - 25, 12)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    K = np.asarray(MagneticsMethod(inducing_field=(50000.0, 60.0, 10.0))
                   .make_simulation_full(mesh, survey).G)
    G = np.asarray(GravityMethod().make_simulation_full(mesh, survey).G)
    return K, G, mesh.to_discretize().cell_centers


def _block(cc, x0, x1, y0, y1, z0=-250, z1=-100):
    return ((cc[:, 0] > x0) & (cc[:, 0] < x1) & (cc[:, 1] > y0) & (cc[:, 1] < y1)
            & (cc[:, 2] > z0) & (cc[:, 2] < z1))


def _dilate(mask):
    """The cells of ``mask`` and their 26 neighbours."""
    m = mask.reshape(NX, NY, NZ, order="F")
    out = m.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out |= np.roll(np.roll(np.roll(m, dx, 0), dy, 1), dz, 2)
    return out.ravel(order="F")


def _data(kernels, beta, rho, rel=0.02, seed=0):
    """Noisy data (2 % of the peak) and their standard deviations."""
    K, G, _ = kernels
    rng = np.random.default_rng(seed)
    out = []
    for A, m in ((K, beta), (G, rho)):
        d = A @ m
        s = rel * max(abs(d).max(), 1e-3)
        out += [d + rng.normal(scale=s, size=d.size), s]
    return out   # f, sigma_f, g, sigma_g


def _n90(r):
    """How many cells hold 90 % of sum ||s_k||^2 (how concentrated the model is)."""
    e = np.sort(np.hypot(r.beta_scaled, r.rho_scaled))[::-1] ** 2
    return int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)


def _jaccard(r, share=0.2):
    """Overlap of the cells where |beta| and |rho| exceed ``share`` of their peaks."""
    b = np.abs(r.beta_physical) > share * np.abs(r.beta_physical).max()
    p = np.abs(r.rho_physical) > share * np.abs(r.rho_physical).max()
    return (b & p).sum() / (b | p).sum()


def _dense(op):
    """The weighted operator as a float64 matrix (tests only)."""
    S = op.S[:, None] if np.ndim(op.S) else op.S
    return S * np.asarray(op.K, dtype=float) * op.w


# ── Group soft threshold ────────────────────────────────────────────────


class TestShrink:
    def test_optimality_of_the_prox(self):
        """s = prox(q) iff mu (q - s) - lambda2 s is lambda1 s/||s|| (s != 0), or within
        lambda1 of the origin (s = 0)."""
        rng = np.random.default_rng(1)
        qb, qr = rng.normal(size=500) * 3, rng.normal(size=500) * 3
        qb[:5] = qr[:5] = 0.0                      # r = 0
        qb[5:10], qr[5:10] = 0.3, -0.2             # below lambda1 / mu
        lam1, lam2, mu = 2.0, 0.7, 4.0
        sb, sr = gl.group_shrink(qb, qr, lam1, lam2, mu)
        r = np.hypot(sb, sr)
        gb, gr = mu * (qb - sb) - lam2 * sb, mu * (qr - sr) - lam2 * sr
        on = r > 0
        np.testing.assert_allclose(gb[on], lam1 * sb[on] / r[on], atol=1e-12)
        np.testing.assert_allclose(gr[on], lam1 * sr[on] / r[on], atol=1e-12)
        assert np.all(np.hypot(gb[~on], gr[~on]) <= lam1 + 1e-12)
        assert not on[:10].any() and on.sum() > 300

    def test_prox_against_a_numerical_minimum(self):
        from scipy.optimize import minimize
        lam1, lam2, mu = 1.5, 0.3, 2.0
        rng = np.random.default_rng(2)
        for q in rng.normal(size=(20, 2)) * 2:
            s = np.array(gl.group_shrink(q[:1], q[1:], lam1, lam2, mu)).ravel()

            def h(v):
                return lam1 * np.hypot(*v) + 0.5 * lam2 * v @ v + 0.5 * mu * (v - q) @ (v - q)
            best = minimize(h, q, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
            assert h(s) <= best.fun + 1e-9

    def test_both_components_shrink_together(self):
        """One factor per cell: the group is not thresholded component by component."""
        qb, qr = np.array([3.0, 0.5, -4.0]), np.array([0.1, 2.0, 4.0])
        sb, sr = gl.group_shrink(qb, qr, lambda1=2.0, lambda2=0.0, mu=1.0)
        np.testing.assert_allclose(sb / qb, sr / qr)
        # the small component of cell 0 survives (0.1 < lambda1 / mu), unlike with
        # separate soft thresholds
        assert sr[0] != 0
        ub, ur = gl.group_shrink(qb, qr, 2.0, 0.0, 1.0, coupling="none")
        assert ur[0] == 0 and ub[1] == 0
        np.testing.assert_allclose(ub, np.sign(qb) * np.maximum(abs(qb) - 2, 0))

    def test_in_place(self):
        qb, qr = np.ones(4), np.zeros(4)
        ob, orr = np.empty(4), np.empty(4)
        rb, rr = gl.group_shrink(qb, qr, 0.5, 0.0, 1.0, ob, orr)
        assert rb is ob and rr is orr
        np.testing.assert_allclose(ob, 0.5)


# ── Scaling ─────────────────────────────────────────────────────────────


class TestScaling:
    def test_weights_give_unit_columns(self, kernels):
        K, G, _ = kernels
        w = gl.sensitivity_weights(K, gamma=2.0)
        np.testing.assert_allclose(np.linalg.norm(K * w, axis=0), 1.0, rtol=1e-5)
        w1 = gl.sensitivity_weights(K, gamma=1.0)
        np.testing.assert_allclose(w1, w ** 0.5, rtol=1e-12)

    def test_column_norms_of_dense_sparse_and_row_scaled(self, kernels):
        K, _, _ = kernels
        s = np.linspace(0.5, 2.0, K.shape[0])
        ref = np.linalg.norm(K.astype(float) * s[:, None], axis=0)
        np.testing.assert_allclose(gl.column_norms(K, s, chunk_bytes=1e5), ref, rtol=1e-6)
        np.testing.assert_allclose(gl.column_norms(sp.csr_matrix(K), s), ref, rtol=1e-6)

    def test_max_ratio(self):
        f, g = np.array([3.0, -120.0]), np.array([0.5, -0.2])
        s_f, s_g, notes = gl.data_scaling(f, g, "max_ratio")
        assert s_f == 1.0 and s_g == pytest.approx(240.0) and not notes

    def test_zero_dataset_is_reported_not_divided_by(self):
        s_f, s_g, notes = gl.data_scaling(np.ones(3), np.zeros(3), "max_ratio")
        assert s_g == 1.0 and "undefined" in notes[0]

    def test_other_modes(self):
        f, g = np.ones(3), np.ones(2)
        s_f, s_g, _ = gl.data_scaling(f, g, "std", std_f=np.full(3, 2.0), std_g=0.5)
        np.testing.assert_allclose(s_f, 0.5)
        np.testing.assert_allclose(s_g, 2.0)
        assert gl.data_scaling(f, g, (2.0, 3.0))[:2] == (2.0, 3.0)
        with pytest.raises(ValueError):
            gl.data_scaling(f, g, "bogus")
        with pytest.raises(ValueError):
            gl.data_scaling(f, g, (1.0, -1.0))

    @pytest.mark.parametrize("scaling", ["max_ratio", "std"])
    def test_physical_models_and_predictions_undo_the_scaling(self, kernels, scaling):
        K, G, cc = kernels
        body = _block(cc, 250, 450, 250, 450)
        f, sf, g, sg = _data(kernels, 0.05 * body, 0.3 * body)
        P = gl.JointGroupLassoProblem(K, G, f, g, data_scaling=scaling,
                                      std_f=np.full(f.size, sf), std_g=np.full(g.size, sg))
        r = P.solve(0.02 * P.lambda1_max(), 1.0)
        np.testing.assert_allclose(r.predicted_magnetic, K @ r.beta_physical, rtol=1e-4,
                                   atol=1e-4 * abs(f).max())
        np.testing.assert_allclose(r.predicted_gravity, G @ r.rho_physical, rtol=1e-4,
                                   atol=1e-4 * abs(g).max())
        np.testing.assert_allclose(r.residual_magnetic, f - r.predicted_magnetic)
        np.testing.assert_allclose(r.residual_gravity, g - r.predicted_gravity)
        # models in the kernels' units: the peaks are of the order of the true 0.05 SI / 0.3 g/cc
        assert 0.01 < r.beta_physical.max() < 0.5 and 0.05 < r.rho_physical.max() < 3.0
        # the scaled variables of the two methods are comparable (the point of the scaling)
        ratio = np.abs(r.rho_scaled).max() / np.abs(r.beta_scaled).max()
        assert 0.1 < ratio < 10


# ── The zeta update ─────────────────────────────────────────────────────


# Accuracy of the zeta solves with a float32 kernel.  One product rounds at ~1e-7
# (relative); the Cholesky factors are made in float64, but CG multiplies by the
# float32 kernel in every iteration, and its rounding accumulates and is amplified
# by the condition of the system: ~1.6e-6 on Apple Silicon (Accelerate), under
# 1e-6 with MKL on Windows.  With a float64 kernel CG reaches ~4e-9.
FLOAT32_SOLVE_TOL = {"cholesky": 1e-6, "cg": 1e-5}


class TestSolvers:
    @pytest.mark.parametrize("cols", [slice(None), slice(0, 60)])   # N < M and N > M
    @pytest.mark.parametrize("solver", ["cholesky", "cg"])
    def test_solves_the_normal_equations(self, kernels, solver, cols):
        K, _, _ = kernels
        K = np.ascontiguousarray(K[:, cols])
        op = gl.WeightedOperator(K, gl.sensitivity_weights(K), row_scale=1.0)
        s = gl.make_solver(op, 3.0, solver)
        X = _dense(op)
        rhs = np.random.default_rng(0).normal(size=X.shape[1])
        x = np.linalg.solve(X.T @ X + 3.0 * np.eye(X.shape[1]), rhs)
        np.testing.assert_allclose(s.solve(rhs, np.zeros_like(rhs)), x, rtol=0,
                                   atol=FLOAT32_SOLVE_TOL[solver] * abs(x).max())
        if solver == "cholesky":
            expect = "data space" if K.shape[0] < K.shape[1] else "model space"
            assert expect in s.name

    def test_per_row_scaling(self, kernels):
        K, _, _ = kernels
        rows = np.linspace(0.5, 2.0, K.shape[0])
        op = gl.WeightedOperator(K, gl.sensitivity_weights(K, row_scale=rows), rows)
        X = _dense(op)
        np.testing.assert_allclose(np.linalg.norm(X, axis=0), 1.0, rtol=1e-5)
        rhs = np.random.default_rng(1).normal(size=X.shape[1])
        x = np.linalg.solve(X.T @ X + 2.0 * np.eye(X.shape[1]), rhs)
        for solver in ("cholesky", "cg"):
            np.testing.assert_allclose(gl.make_solver(op, 2.0, solver).solve(rhs, None), x,
                                       rtol=0, atol=FLOAT32_SOLVE_TOL[solver] * abs(x).max())

    def test_auto_picks_cg_when_the_factor_is_too_big(self, kernels):
        K, _, _ = kernels
        op = gl.WeightedOperator(K, gl.sensitivity_weights(K))
        assert "cholesky" in gl.make_solver(op, 1.0).name
        assert gl.make_solver(op, 1.0, factor_max_bytes=1e3).name == "conjugate gradients"

    def test_float32_kernels_are_not_copied_to_float64(self, kernels):
        K, _, _ = kernels
        assert K.dtype == np.float32
        op = gl.WeightedOperator(K, np.ones(K.shape[1]))
        assert op.K is K and op.matvec(np.ones(K.shape[1])).dtype == np.float64


# ── ADMM ────────────────────────────────────────────────────────────────


def _fista(P, lam1, lam2, n_iter=6000):
    """Accelerated proximal gradient on the same objective: an independent reference."""
    X, Y = _dense(P.X), _dense(P.Y)
    L = max(np.linalg.norm(X, 2), np.linalg.norm(Y, 2)) ** 2 + lam2
    b = np.zeros(P.m)
    r = np.zeros(P.m)
    yb, yr, t = b.copy(), r.copy(), 1.0
    for _ in range(n_iter):
        gb = X.T @ (X @ yb - P.bf) + lam2 * yb
        gr = Y.T @ (Y @ yr - P.bg) + lam2 * yr
        nb, nr = gl.group_shrink(yb - gb / L, yr - gr / L, lam1 / L, 0.0, 1.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        yb, yr = nb + (t - 1) / t_new * (nb - b), nr + (t - 1) / t_new * (nr - r)
        b, r, t = nb, nr, t_new
    return b, r


@pytest.fixture(scope="module")
def coincident(kernels):
    K, G, cc = kernels
    body = _block(cc, 250, 450, 250, 450)
    f, sf, g, sg = _data(kernels, 0.05 * body, 0.3 * body)
    return gl.JointGroupLassoProblem(K, G, f, g), body, (sf, sg)


@pytest.fixture(scope="module")
def coincident64(kernels, coincident):
    """The same problem with float64 kernels: products with float32 kernels leave
    the ADMM residuals a floor of ~1e-6 (relative), fine for the default 1e-4
    tolerances but not for checks at 1e-7."""
    K, G, _ = kernels
    P = coincident[0]
    return gl.JointGroupLassoProblem(K.astype(float), G.astype(float), P.f, P.g)


def _objective(P, b, r, lam1, lam2):
    t = P.terms(b, r)
    return t["misfit"] + lam1 * t["group"] + lam2 * t["l2"]


class TestADMM:
    # pure group lasso (lambda2 = 0) converges slowly; a smaller mu helps it
    @pytest.mark.parametrize("lam2, mu", [(1.0, None), (0.0, 2.0)])
    def test_reaches_the_minimizer(self, kernels, coincident64, lam2, mu):
        P = coincident64
        if mu is not None:
            K, G, _ = kernels
            P = gl.JointGroupLassoProblem(K.astype(float), G.astype(float), P.f, P.g, mu=mu)
        lam1 = 0.03 * P.lambda1_max()
        r = P.solve(lam1, lam2, tol_primal=1e-7, tol_dual=1e-7, max_iter=30000)
        assert r.converged
        assert P.kkt_residual(r.beta_scaled, r.rho_scaled, lam1, lam2) < 1e-4
        # an independent solver of the same objective gets no lower
        fb, fr = _fista(P, lam1, lam2)
        assert r.objective <= _objective(P, fb, fr, lam1, lam2) * (1 + 1e-7)
        if lam2 > 0:   # strongly convex: one minimizer
            scale = np.linalg.norm(np.concatenate([fb, fr]))
            assert np.linalg.norm(np.concatenate([r.beta_scaled - fb, r.rho_scaled - fr])) \
                < 1e-3 * scale

    def test_mu_changes_the_path_not_the_minimizer(self, kernels, coincident64):
        P = coincident64
        lam1 = 0.03 * P.lambda1_max()
        other = gl.JointGroupLassoProblem(P.X.K, P.Y.K, P.f, P.g, mu=P.mu / 5)
        a = P.solve(lam1, 0.5, tol_primal=1e-7, tol_dual=1e-7, max_iter=30000)
        b = other.solve(lam1, 0.5, tol_primal=1e-7, tol_dual=1e-7, max_iter=30000)
        assert a.converged and b.converged and a.n_iterations != b.n_iterations
        np.testing.assert_allclose(b.beta_scaled, a.beta_scaled, atol=1e-4 * abs(a.beta_scaled).max())
        np.testing.assert_allclose(b.rho_scaled, a.rho_scaled, atol=1e-4 * abs(a.rho_scaled).max())

    def test_default_tolerances_are_close_to_optimal(self, coincident):
        P, _, _ = coincident
        lam1 = 0.03 * P.lambda1_max()
        r = P.solve(lam1, 1.0)
        assert r.converged and P.kkt_residual(r.beta_scaled, r.rho_scaled, lam1, 1.0) < 1e-2

    def test_solvers_and_precisions_agree(self, kernels, coincident, coincident64):
        P = coincident64
        lam1 = 0.03 * P.lambda1_max()
        ref = P.solve(lam1, 1.0, tol_primal=1e-7, tol_dual=1e-7, max_iter=20000)
        assert ref.converged
        K, G, _ = (np.asarray(k, dtype=float) if i < 2 else k for i, k in enumerate(kernels))
        cg_problem = gl.JointGroupLassoProblem(K, G, P.f, P.g, mu=P.mu, solver="cg")
        # matrix-free operators, with their weights given
        free = gl.JointGroupLassoProblem(aslinearoperator(K), aslinearoperator(G), P.f, P.g,
                                         mu=P.mu, magnetic_weights=P.w_beta,
                                         gravity_weights=P.w_rho)
        assert cg_problem.solver_name == free.solver_name == "conjugate gradients"
        for Q in (cg_problem, free):
            r = Q.solve(lam1, 1.0, tol_primal=1e-7, tol_dual=1e-7, max_iter=20000)
            np.testing.assert_allclose(r.beta_physical, ref.beta_physical,
                                       atol=1e-5 * abs(ref.beta_physical).max())
            np.testing.assert_allclose(r.rho_physical, ref.rho_physical,
                                       atol=1e-5 * abs(ref.rho_physical).max())
        # float32 kernels (as SimPEG stores them) at the default tolerances
        r32 = coincident[0].solve(lam1, 1.0)
        np.testing.assert_allclose(r32.beta_physical, ref.beta_physical,
                                   atol=2e-3 * abs(ref.beta_physical).max())
        np.testing.assert_allclose(r32.rho_physical, ref.rho_physical,
                                   atol=2e-3 * abs(ref.rho_physical).max())

    def test_zero_above_lambda1_max(self, coincident):
        P, _, _ = coincident
        top = P.lambda1_max()
        r = P.solve(top * 1.0001, 0.3)
        assert r.n_iterations == 0 and r.converged and r.n_active == 0
        assert not np.any(r.beta_scaled) and not np.any(r.rho_scaled)
        assert P.solve(top * 0.9, 0.3).n_active > 0

    def test_histories_and_residual_stopping(self, coincident):
        P, _, _ = coincident
        seen = []
        r = P.solve(0.03 * P.lambda1_max(), 1.0, callback=lambda i, rec: seen.append(rec))
        n = r.n_iterations
        for h in (r.objective_history, r.misfit_history, r.group_penalty_history,
                  r.l2_penalty_history, r.primal_residual_history, r.dual_residual_history):
            assert len(h) == n
        last = seen[-1]
        assert last["primal"] <= last["eps_primal"] and last["dual"] <= last["eps_dual"]
        np.testing.assert_allclose(r.objective_history,
                                   np.array(r.misfit_history) + r.lambda1 * np.array(r.group_penalty_history)
                                   + r.lambda2 * np.array(r.l2_penalty_history))
        # too few iterations: not converged, but the iterate is returned
        short = P.solve(0.03 * P.lambda1_max(), 1.0, max_iter=3)
        assert short.n_iterations == 3 and not short.converged
        assert len(P.solve(0.03 * P.lambda1_max(), 1.0, history_every=10).misfit_history) < n

    def test_stop_keeps_the_iterate(self, coincident):
        P, _, _ = coincident
        calls = []
        r = P.solve(0.03 * P.lambda1_max(), 1.0,
                    should_stop=lambda: calls.append(1) or len(calls) > 5)
        assert r.stopped and not r.converged and r.n_iterations == 5
        assert np.all(np.isfinite(r.beta_physical))

    def test_warm_start_saves_iterations(self, coincident):
        P, _, _ = coincident
        lam1 = 0.03 * P.lambda1_max()
        first = P.solve(lam1 * 1.3, 1.0)
        cold = P.solve(lam1, 1.0)
        warm = P.solve(lam1, 1.0, state=first.state)
        assert warm.n_iterations < cold.n_iterations

    def test_convenience_function(self, kernels, coincident):
        K, G, _ = kernels
        P, _, _ = coincident
        lam1 = 0.03 * P.lambda1_max()
        r = gl.joint_group_lasso_admm(magnetic_operator=K, gravity_operator=G, magnetic_data=P.f,
                                      gravity_data=P.g, lambda1=lam1, lambda2=1.0, mu=P.mu,
                                      max_iter=5000, tol_primal=1e-4, tol_dual=1e-4)
        np.testing.assert_allclose(r.beta_physical, P.solve(lam1, 1.0).beta_physical)

    def test_rejects_mismatched_input(self, kernels):
        K, G, _ = kernels
        with pytest.raises(ValueError, match="same cells"):
            gl.JointGroupLassoProblem(K, G[:, :10], np.ones(K.shape[0]), np.ones(G.shape[0]))
        with pytest.raises(ValueError, match="rows"):
            gl.JointGroupLassoProblem(K, G, np.ones(3), np.ones(G.shape[0]))
        with pytest.raises(ValueError, match="mu"):
            gl.JointGroupLassoProblem(K, G, np.ones(K.shape[0]), np.ones(G.shape[0]), mu=0.0)


# ── Synthetic validation (A–E) ──────────────────────────────────────────


class TestSynthetics:
    def test_a_coincident_body_recovered_colocated(self, coincident):
        """Both models sit on the same cells, and closer together than with separate
        (ordinary elastic-net) sparsity on each model."""
        P, body, _ = coincident
        lc = P.lcurve(1.0)
        assert lc.corner["valid"]
        r = P.solve_at(lc, lc.lambda1_corner, 1.0)
        near = _dilate(body)
        for v in (r.beta_physical, r.rho_physical):
            assert np.abs(v)[near].sum() > 0.7 * np.abs(v).sum()
        separate = P.solve(r.lambda1, 1.0, coupling="none", max_iter=5000)
        assert _jaccard(r) > 0.5
        assert _jaccard(r) > _jaccard(separate) + 0.1

    def test_b_magnetic_only_body_gets_no_density(self, kernels):
        """A cell selected for its magnetization keeps rho ~ 0: the group lasso does
        not ask for both properties."""
        K, G, cc = kernels
        body, mag_only = _block(cc, 150, 350, 150, 350), _block(cc, 500, 700, 500, 700)
        f, _, g, _ = _data(kernels, 0.05 * (body | mag_only), 0.3 * body)
        P = gl.JointGroupLassoProblem(K, G, f, g)
        r = P.solve(0.02 * P.lambda1_max(), 1.0)
        b, p = np.abs(r.beta_physical), np.abs(r.rho_physical)
        assert b[mag_only].mean() > 0.5 * b[body].mean()
        assert p[mag_only].mean() < 0.15 * p[body].mean()

    def test_b_without_gravity_signal_rho_stays_exactly_zero(self, kernels):
        K, G, cc = kernels
        body = _block(cc, 250, 450, 250, 450)
        f = K @ (0.05 * body)
        g = np.zeros(G.shape[0])
        with pytest.warns(UserWarning, match="undefined"):
            P = gl.JointGroupLassoProblem(K, G, f, g)
        r = P.solve(0.02 * P.lambda1_max(), 1.0)
        assert r.n_active > 0 and not np.any(r.rho_scaled)
        assert "undefined" in r.notes[0]

    def test_c_gravity_only_body_gets_no_magnetization(self, kernels):
        K, G, cc = kernels
        body, grav_only = _block(cc, 150, 350, 150, 350), _block(cc, 500, 700, 500, 700)
        f, _, g, _ = _data(kernels, 0.05 * body, 0.3 * (body | grav_only))
        P = gl.JointGroupLassoProblem(K, G, f, g)
        r = P.solve(0.02 * P.lambda1_max(), 1.0)
        b, p = np.abs(r.beta_physical), np.abs(r.rho_physical)
        assert p[grav_only].mean() > 0.5 * p[body].mean()
        assert b[grav_only].mean() < 0.15 * b[body].mean()

    def test_d_opposite_signs(self, kernels):
        """The group norm does not care about signs: beta > 0 with rho < 0."""
        K, G, cc = kernels
        body = _block(cc, 250, 450, 250, 450)
        f, _, g, _ = _data(kernels, 0.05 * body, -0.3 * body)
        P = gl.JointGroupLassoProblem(K, G, f, g)
        r = P.solve(0.02 * P.lambda1_max(), 1.0)
        gn = np.hypot(r.beta_scaled, r.rho_scaled)
        core = gn > 0.3 * gn.max()
        assert core.sum() >= 10
        assert np.all(r.beta_physical[core] > 0) and np.all(r.rho_physical[core] < 0)
        assert np.corrcoef(r.beta_physical, r.rho_physical)[0, 1] < -0.8
        assert _jaccard(r) > 0.5

    def test_e_l2_moderates_the_group_lasso(self, kernels, coincident):
        """Group lasso alone concentrates the model into a few cells with inflated
        values; lambda2 spreads it; L2 alone has no sparsity at all."""
        P, body, _ = coincident
        lam1 = 0.01 * P.lambda1_max()
        l2_only = P.solve(0.0, 1.0)
        both = P.solve(lam1, 1.0)
        # lambda2 = 0 converges slowly (see TestADMM); a smaller mu, and near-optimality
        # checked on the optimality conditions
        K, G, _ = kernels
        Q = gl.JointGroupLassoProblem(K, G, P.f, P.g, mu=2.0)
        gl_only = Q.solve(lam1, 0.0, max_iter=20000)
        assert Q.kkt_residual(gl_only.beta_scaled, gl_only.rho_scaled, lam1, 0.0) < 2e-3
        assert both.converged and l2_only.converged
        assert l2_only.n_active == P.m
        assert _n90(gl_only) < _n90(both) < _n90(l2_only)
        assert gl_only.n_active < both.n_active
        # the true body is 0.3 g/cc: alone the group lasso inflates it several times
        assert gl_only.rho_physical.max() > 3 * both.rho_physical.max()


class TestFromSimulations:
    def test_uses_simpeg_sensitivities_as_stored(self, kernels):
        from geoinv3d.datamodel.mesh import Mesh3D
        from geoinv3d.datamodel.survey import SurveyData
        from geoinv3d.methods.gravity import GravityMethod
        from geoinv3d.methods.magnetics import MagneticsMethod

        mesh = Mesh3D.uniform(8, 8, 4, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -200.0))
        xy = np.linspace(25, 375, 6)
        xx, yy = np.meshgrid(xy, xy)
        locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)])
        survey = SurveyData(locations=locs, observed=np.zeros(36), std=np.ones(36))
        msim = MagneticsMethod(inducing_field=(50000.0, 60.0, 10.0)).make_simulation_full(mesh, survey)
        gsim = GravityMethod().make_simulation_full(mesh, survey)
        m = np.zeros(mesh.n_cells)
        m[100:110] = 1.0
        f, g = msim.dpred(0.05 * m), gsim.dpred(0.3 * m)
        P = gl.from_simulations(msim, gsim, f, g)
        assert P.X.K is msim.G and P.Y.K is gsim.G
        r = P.solve(0.05 * P.lambda1_max(), 1.0)
        np.testing.assert_allclose(r.predicted_magnetic, msim.dpred(r.beta_physical),
                                   rtol=1e-4, atol=1e-4 * abs(f).max())
        np.testing.assert_allclose(r.predicted_gravity, gsim.dpred(r.rho_physical),
                                   rtol=1e-4, atol=1e-4 * abs(g).max())


class TestLCurve:
    def test_sweep_and_corner(self, coincident, tmp_path):
        P, _, _ = coincident
        seen = []
        lc = P.lcurve(1.0, n_lambda1=13, decades=3.0,
                      point_callback=lambda i, n, p: seen.append((i, n)))
        a = lc.arrays()
        assert len(a["lambda1"]) == 13 == len(seen) and seen[-1] == (12, 13)
        assert np.all(np.diff(a["lambda1"]) < 0)
        # smaller lambda1: better fit, larger group penalty
        assert np.all(np.diff(a["misfit"]) < 0) and np.all(np.diff(a["group_penalty"]) > 0)
        assert a["lambda1"].min() < lc.lambda1_corner < a["lambda1"].max()
        # warm starts: later points need fewer iterations than the first
        assert a["n_iterations"][-1] < a["n_iterations"][0]
        import matplotlib
        matplotlib.use("Agg")
        ax = gl.plot_lcurve(lc)
        ax.figure.savefig(tmp_path / "lcurve.png")
        assert (tmp_path / "lcurve.png").stat().st_size > 1000

    def test_stopped_sweep_keeps_its_points(self, coincident):
        P, _, _ = coincident
        calls = []
        lc = P.lcurve(1.0, n_lambda1=13, should_stop=lambda: calls.append(1) or len(calls) > 400)
        assert lc.stopped and 0 < len(lc.points) < 13 and lc.n_planned == 13


# ── Through the data pipeline (worker "data" mode, run locally) ───────────


def _pipeline_files(tmp_path):
    from tests.test_data_pipeline import _station_grid, _synthetic, _write_csv
    locs = _station_grid()
    _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
    _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))
    return tmp_path


def _progress(seen):
    """A progress callback that keeps the per-point reports of the inversion."""
    def progress(stage, message="", **fields):
        if stage == "inverting" and "iteration" in fields:
            seen.append(fields)
    return progress


class TestPipeline:
    def test_joint_group_lasso_end_to_end(self, tmp_path):
        import json
        import zipfile

        from tests.test_data_pipeline import _joint_params
        from geoinv3d.cloud.worker import pack_result, run_data_pipeline
        from geoinv3d.viz.result_workflow import build_workflow, load_result

        _pipeline_files(tmp_path)
        params = _joint_params(["g.csv"], ["m.csv"], param_mode="manual",
                               regularization_type="group_lasso", gl_n_lambda1=8,
                               gl_lambda1_decades=2.5, gl_lambda2=0.3)
        seen = []
        result = run_data_pipeline(params, str(tmp_path), progress=_progress(seen))
        assert result["regularization"] == "group_lasso_ADMM"
        assert result["methods"] == ["gravity", "magnetics"]
        n = result["n_active_cells"]
        assert result["recovered_models"]["gravity"].shape == (n,)
        assert result["recovered_models"]["magnetics"].shape == (n,)
        info = result["group_lasso"]
        assert info["criterion"] in ("lcurve", "discrepancy", "nearest chi^2 = N")
        assert len(info["sweep"]) == 8 == result["n_iterations"]
        lams = [it["beta"] for it in result["iterations"]]
        assert lams == sorted(lams, reverse=True) and lams[-1] <= info["lambda1"] <= lams[0]
        # progress: one report per lambda1 point, "i of n", with chi^2 against N
        assert [s["iteration"] for s in seen] == list(range(1, 9))
        assert all(s["max_iter"] == 8 for s in seen)
        np.testing.assert_allclose([s["phi_d"] for s in seen],
                                   [p["chi2"] for p in info["sweep"]])
        # per-dataset data in the files' convention (gravity positive down)
        jd = result["joint_data"]
        from tests.test_data_pipeline import _station_grid, _synthetic
        np.testing.assert_allclose(jd["gravity"]["observed"],
                                   _synthetic("gravity", _station_grid()), rtol=1e-6)
        for name in ("gravity", "magnetics"):
            d = jd[name]
            r = d["observed"] - d["predicted"]
            assert np.sqrt(np.mean(r ** 2)) < 0.5 * np.sqrt(np.mean(d["observed"] ** 2))
        assert result["settings"]["regularization_type"] == "group_lasso"
        assert result["settings"]["gl_lambda2"] == 0.3

        path = pack_result(result, str(tmp_path / "result.zip"))
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            meta = json.loads(zf.read("result.json"))
        assert {"recovered_gravity.npy", "recovered_magnetics.npy", "data_gravity.npz",
                "data_magnetics.npz"} <= names
        assert "joint_data" not in meta and meta["group_lasso"]["lambda1"] == info["lambda1"]

        # the viewer shows it as a density and a susceptibility model, each with its data
        run = load_result(path)
        run["_name"] = "joint"
        wf = build_workflow([run])
        inv = [nd for nd in wf["nodes"] if nd["type"] == "RegularizedInversionNode"]
        assert sorted(nd["output"]["final_model"]["prop"] for nd in inv) == ["density", "susceptibility"]
        assert {nd["output"]["data_fit"]["unit"] for nd in inv} == {"mGal", "nT"}
        assert all(nd["params"]["gl_criterion"] == info["criterion"] for nd in inv)
        surveys = [nd for nd in wf["nodes"] if nd["type"] == "SurveyCreateNode"]
        assert len(surveys) == 2
        assert any(nd["name"] == "λ2 = 0.3" for nd in wf["nodes"])

    def test_auto_mode_and_stop(self, tmp_path):
        from tests.test_data_pipeline import _joint_params
        from geoinv3d.cloud.worker import run_data_pipeline
        from geoinv3d.methods.directives import IterationCollector

        _pipeline_files(tmp_path)
        # auto mode ignores the manual gl_ settings but keeps the regularization
        params = _joint_params(["g.csv"], ["m.csv"], regularization_type="group_lasso",
                               gl_n_lambda1=5)
        seen = []
        IterationCollector.stop_check = lambda: len(seen) >= 4   # "stop & keep" after 4 points
        try:
            result = run_data_pipeline(params, str(tmp_path), progress=_progress(seen))
        finally:
            IterationCollector.stop_check = None
        assert result["regularization"] == "group_lasso_ADMM"
        assert result["stopped_early"] == {"reason": "stopped by the user", "at_iteration": 4,
                                           "of": 13}
        assert result["converged"] is False and len(result["group_lasso"]["sweep"]) == 4
        assert any("Stopped by the user" in w for w in result["group_lasso"]["warnings"])
        assert np.all(np.isfinite(result["recovered_models"]["gravity"]))

    def test_two_gravity_datasets_of_one_density_model(self, tmp_path):
        """Upload-page datasets with a "model" label share it; the viewer shows the model
        with its first dataset."""
        from tests.test_data_pipeline import _joint_params, _station_grid, _synthetic, _write_csv
        from geoinv3d.cloud.worker import pack_result, run_data_pipeline
        from geoinv3d.viz.result_workflow import build_workflow, load_result

        _pipeline_files(tmp_path)
        locs = _station_grid() + np.array([50.0, 50.0, 0.0])      # a second, offset survey
        _write_csv(tmp_path / "g2.csv", locs, _synthetic("gravity", locs))
        params = _joint_params(["g.csv"], ["m.csv"], param_mode="manual",
                               regularization_type="group_lasso", gl_n_lambda1=5,
                               gl_lambda1_decades=2.0)
        second = dict(params["datasets"][0], files=["g2.csv"])
        params["datasets"].insert(1, second)
        params["datasets"][0]["model"] = params["datasets"][1]["model"] = "density"
        result = run_data_pipeline(params, str(tmp_path))
        assert set(result["recovered_models"]) == {"density", "magnetics"}
        assert result["dataset_labels"] == ["gravity", "gravity_2", "magnetics"]
        assert set(result["joint_data"]) == {"gravity", "gravity_2", "magnetics"}
        assert result["group_lasso"]["models"]["density"] == ["gravity", "gravity_2"]
        run = load_result(pack_result(result, str(tmp_path / "result.zip")))
        run["_name"] = "joint"
        inv = [nd for nd in build_workflow([run])["nodes"]
               if nd["type"] == "RegularizedInversionNode"]
        assert sorted(nd["output"]["final_model"]["prop"] for nd in inv) == \
            ["density", "susceptibility"]

    def test_needs_a_gravity_and_a_magnetic_dataset(self, tmp_path):
        from tests.test_data_pipeline import _single
        from geoinv3d.cloud.worker import run_data_pipeline
        _pipeline_files(tmp_path)
        params = _single("gravity", ["g.csv"], regularization_type="group_lasso")
        with pytest.raises(ValueError, match="joint inversion"):
            run_data_pipeline(params, str(tmp_path))

    def test_task_round_trip(self, tmp_path):
        from geoinv3d.cloud.task import InversionTask, pack_task, unpack_task
        task = InversionTask(task_id="t", regularization_type="group_lasso", gl_lambda2=0.3,
                             gl_mu=5.0, gl_lambda1_selection="fixed", gl_lambda1=12.0)
        back = unpack_task(pack_task(task, str(tmp_path / "t.zip")))
        assert (back.gl_lambda2, back.gl_mu, back.gl_lambda1_selection, back.gl_lambda1) == \
            (0.3, 5.0, "fixed", 12.0)


def test_workflow_skips_joint_runs_without_data(tmp_path):
    """A cross-gradient joint result (no per-dataset data) no longer breaks a workflow."""
    from geoinv3d.viz.result_workflow import split_joint_runs
    runs, skipped = split_joint_runs([{"_name": "xgrad", "_model": None,
                                       "_models": {"gravity": np.zeros(3)}}])
    assert runs == [] and skipped == ["xgrad"]


# ── Generalization: P models, several datasets, cross-gradient, nonlinear ──


class TestStackedOperator:
    def test_matches_the_dense_stack(self, kernels):
        K, G, _ = kernels
        w = np.random.default_rng(0).uniform(0.5, 2.0, K.shape[1])
        s2 = np.linspace(0.5, 2.0, G.shape[0])
        op = gl.StackedOperator([gl.WeightedOperator(K, w, 3.0), gl.WeightedOperator(G, w, s2)])
        dense = np.vstack([3.0 * K.astype(float), s2[:, None] * G.astype(float)]) * w
        v, y = np.random.default_rng(1).normal(size=(2, K.shape[1])), None
        y = np.random.default_rng(2).normal(size=op.shape[0])
        # float32 kernels: products round at ~1e-7 of their magnitude
        ref = dense @ v[0]
        np.testing.assert_allclose(op.matvec(v[0]), ref, rtol=0, atol=1e-5 * abs(ref).max())
        ref = dense.T @ y
        np.testing.assert_allclose(op.rmatvec(y), ref, rtol=0, atol=1e-5 * abs(ref).max())
        np.testing.assert_allclose(np.triu(op.gram()), np.triu(dense @ dense.T), rtol=1e-5,
                                   atol=1e-6 * abs(dense @ dense.T).max())
        np.testing.assert_allclose(np.triu(op.normal()), np.triu(dense.T @ dense), rtol=1e-5,
                                   atol=1e-6 * abs(dense.T @ dense).max())
        np.testing.assert_allclose(op.diag_normal(), np.sum(dense ** 2, axis=0), rtol=1e-5)
        with pytest.raises(ValueError, match="column weights"):
            gl.StackedOperator([gl.WeightedOperator(K, w), gl.WeightedOperator(G, 2 * w)])


def test_group_shrink_of_three_models():
    rng = np.random.default_rng(3)
    qs = [rng.normal(size=300) * 2 for _ in range(3)]
    qs[0][:5] = qs[1][:5] = qs[2][:5] = 0.0
    lam1, lam2, mu = 1.5, 0.4, 3.0
    s = gl.group_shrink_many(qs, lam1, lam2, mu)
    r = np.sqrt(sum(x ** 2 for x in s))
    g = [mu * (q - x) - lam2 * x for q, x in zip(qs, s)]
    on = r > 0
    for gp, x in zip(g, s):
        np.testing.assert_allclose(gp[on], lam1 * x[on] / r[on], atol=1e-12)
    assert np.all(np.sqrt(sum(gp[~on] ** 2 for gp in g)) <= lam1 + 1e-12)
    # two models: the same as group_shrink
    a = gl.group_shrink_many(qs[:2], lam1, lam2, mu)
    b = gl.group_shrink(qs[0], qs[1], lam1, lam2, mu)
    np.testing.assert_array_equal(a[0], b[0])


class TestCrossGradient:
    def test_matches_simpeg(self):
        from simpeg import maps
        from simpeg.regularization import CrossGradient
        mesh = Mesh3D_uniform()
        n = mesh.nC
        rng = np.random.default_rng(4)
        u1, u2, v = rng.normal(size=(3, n))
        t = gl.CrossGradientTerm(mesh)
        ref = CrossGradient(mesh, wire_map=maps.Wires(("a", n), ("b", n)))
        assert t.value(u1, u2) == pytest.approx(ref(np.r_[u1, u2]), rel=1e-12)
        np.testing.assert_allclose(t.gradient(u1, u2), ref.deriv(np.r_[u1, u2])[:n], rtol=1e-10,
                                   atol=1e-12 * abs(t.gradient(u1, u2)).max())
        Q = t.quadratic(u2)
        assert u1 @ Q(u1) == pytest.approx(t.value(u1, u2), rel=1e-12)
        assert v @ Q(u1) == pytest.approx(u1 @ Q(v), rel=1e-10)          # symmetric
        assert all(x @ Q(x) >= -1e-9 for x in rng.normal(size=(5, n)))   # PSD

    def test_kkt_and_monotone_coupling(self, small):
        """At convergence the conditions of the whole (nonconvex) objective hold, and a
        larger lambda3 gives better aligned models."""
        K, G, f, g, mesh = small
        P = gl.JointGroupLassoProblem(K, G, f, g, mesh=mesh)
        lam1 = 0.03 * P.lambda1_max()
        values = []
        for lam3 in (0.0, 0.03, 0.3):
            r = P.solve(lam1, 0.3, cross_gradient=lam3, tol_primal=1e-7, tol_dual=1e-7,
                        max_iter=30000)
            assert r.converged
            # the Gauss–Seidel zeta update leaves a floor of ~1e-4; without the
            # cross-gradient's gradient the conditions would be violated far more
            assert P.kkt_residual(r.beta_scaled, r.rho_scaled, lam1, 0.3, lam3) < 1e-3
            if lam3 > 0:
                assert P.kkt_residual(r.beta_scaled, r.rho_scaled, lam1, 0.3, 0.0) > 1e-2
            values.append(P.cross_gradient_value(r.models_scaled))
            np.testing.assert_allclose(r.objective_history[-1], r.misfit_history[-1]
                                       + lam1 * r.group_penalty_history[-1]
                                       + 0.3 * r.l2_penalty_history[-1]
                                       + lam3 * r.cross_gradient_history[-1], rtol=1e-10)
        assert values[0] > values[1] > values[2]
        # zero is still the solution above lambda1_max
        assert P.solve(1.001 * P.lambda1_max(), 0.3, cross_gradient=0.3).n_active == 0
        with pytest.raises(ValueError, match="mesh"):
            gl.JointGroupLassoProblem(K, G, f, g).solve(lam1, 0.3, cross_gradient=0.1)

    def test_weight_is_unit_free(self, small):
        """lambda3 means the same whatever the units of the models and the data."""
        K, G, f, g, mesh = small
        tight = dict(tol_primal=1e-8, tol_dual=1e-8, max_iter=30000)
        runs = []
        for Gu, gu in ((G, g), (G / 1000.0, g), (1000.0 * G, 1000.0 * g)):
            P = gl.JointGroupLassoProblem(K, Gu, f, gu, mesh=mesh)
            runs.append(P.solve(0.03 * P.lambda1_max(), 0.3, cross_gradient=0.1, **tight))
        a, b, c = runs
        for r in (b, c):
            np.testing.assert_allclose(r.rho_scaled, a.rho_scaled, rtol=0,
                                       atol=1e-5 * abs(a.rho_scaled).max())
            assert r.final_terms["cross_gradient"] == pytest.approx(
                a.final_terms["cross_gradient"], rel=1e-5)
        np.testing.assert_allclose(b.rho_physical, 1000.0 * a.rho_physical, rtol=0,
                                   atol=1e-5 * abs(b.rho_physical).max())


@pytest.fixture(scope="module")
def small():
    """Magnetic and gravity kernels (float64) of 8 x 8 x 6 cells, and noisy data of a block."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod
    mesh = Mesh3D.uniform(8, 8, 6, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -300.0))
    xy = np.linspace(25, 375, 8)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    sv = SurveyData(locations=locs, observed=np.zeros(64), std=np.ones(64))
    K = np.asarray(MagneticsMethod(inducing_field=(50000.0, 60.0, 10.0))
                   .make_simulation_full(mesh, sv).G, dtype=float)
    G = np.asarray(GravityMethod().make_simulation_full(mesh, sv).G, dtype=float)
    cc = mesh.to_discretize().cell_centers
    body = (cc[:, 0] > 100) & (cc[:, 0] < 250) & (cc[:, 1] > 100) & (cc[:, 1] < 250) \
        & (cc[:, 2] > -200) & (cc[:, 2] < -50)
    rng = np.random.default_rng(6)
    f, g = K @ (0.05 * body), G @ (0.3 * body)
    f, g = (d + rng.normal(scale=0.02 * abs(d).max(), size=d.size) for d in (f, g))
    return K, G, f, g, mesh.to_discretize()


def Mesh3D_uniform():
    from geoinv3d.datamodel.mesh import Mesh3D
    return Mesh3D.uniform(NX, NY, NZ, H, H, H, origin=(0.0, 0.0, -NZ * H)).to_discretize()


class TestGeneralProblem:
    def test_datasets_of_one_model_stack(self, kernels):
        """gz and gzz of one density model = one stacked gravity operator."""
        from geoinv3d.datamodel.mesh import Mesh3D
        from geoinv3d.datamodel.survey import SurveyData
        from geoinv3d.methods.gravity import GravityMethod
        K, G, cc = kernels
        mesh = Mesh3D.uniform(NX, NY, NZ, H, H, H, origin=(0.0, 0.0, -NZ * H))
        xy = np.linspace(25, NX * H - 25, 12)
        xx, yy = np.meshgrid(xy, xy)
        locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
        Gzz = np.asarray(GravityMethod("gzz").make_simulation_full(
            mesh, SurveyData(locations=locs, observed=np.zeros(144), std=np.ones(144))).G)
        body = _block(cc, 250, 450, 250, 450)
        f, sf, g, sg = _data(kernels, 0.05 * body, 0.3 * body)
        gzz = Gzz @ (0.3 * body)
        szz = 0.02 * abs(gzz).max()
        gzz = gzz + np.random.default_rng(5).normal(scale=szz, size=gzz.size)
        std = dict(data_scaling="std")
        K, G, Gzz = (np.asarray(a, dtype=float) for a in (K, G, Gzz))   # for a tight check
        P = gl.GroupLassoProblem([gl.GroupLassoData("magnetic", f, 0, sf, operator=K),
                                  gl.GroupLassoData("gz", g, 1, sg, operator=G),
                                  gl.GroupLassoData("gzz", gzz, 1, szz, operator=Gzz)],
                                 model_names=["susceptibility", "density"], **std)
        assert isinstance(P.ops[1], gl.StackedOperator)
        Q = gl.JointGroupLassoProblem(K, np.vstack([G, Gzz]), f, np.r_[g, gzz],
                                      std_f=sf, std_g=np.r_[np.full(144, sg), np.full(144, szz)],
                                      **std)
        assert P.mu == pytest.approx(Q.mu) and P.lambda1_max() == pytest.approx(Q.lambda1_max())
        lam1 = 0.03 * P.lambda1_max()
        tight = dict(tol_primal=1e-8, tol_dual=1e-8, max_iter=30000)
        a, b = P.solve(lam1, 0.3, **tight), Q.solve(lam1, 0.3, **tight)
        np.testing.assert_allclose(a.model("density"), b.rho_physical, rtol=0,
                                   atol=1e-5 * abs(b.rho_physical).max())
        chi2 = P.chi2_of(a)
        assert set(chi2) == {"magnetic", "gz", "gzz", "total"}
        assert a.prediction("gzz").shape == (144,)

    def test_linear_data_through_the_gauss_newton_path(self, kernels):
        """A linear method given as a simulation takes the nonlinear path and gets the
        same answer (the linearization is exact) in few steps."""
        from geoinv3d.datamodel.mesh import Mesh3D
        from geoinv3d.datamodel.survey import SurveyData
        from geoinv3d.methods.gravity import GravityMethod
        K, G, cc = kernels
        mesh = Mesh3D.uniform(NX, NY, NZ, H, H, H, origin=(0.0, 0.0, -NZ * H))
        xy = np.linspace(25, NX * H - 25, 12)
        xx, yy = np.meshgrid(xy, xy)
        locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
        sim = GravityMethod().make_simulation_full(
            mesh, SurveyData(locations=locs, observed=np.zeros(144), std=np.ones(144)))
        body = _block(cc, 250, 450, 250, 450)
        f, sf, g, sg = _data(kernels, 0.05 * body, 0.3 * body)
        make = lambda grav: gl.GroupLassoProblem(
            [gl.GroupLassoData("magnetic", f, 0, sf, operator=K), grav], data_scaling="std")
        lin = make(gl.GroupLassoData("gravity", g, 1, sg, operator=G))
        nl = make(gl.GroupLassoData("gravity", g, 1, sg, simulation=sim))
        assert lin.linear and not nl.linear
        lam1 = 0.03 * lin.lambda1_max()
        a = lin.solve(lam1, 0.3, tol_primal=1e-6, tol_dual=1e-6, max_iter=20000)
        b = nl.solve(lam1, 0.3, tol_primal=1e-6, tol_dual=1e-6, max_iter=20000)
        assert b.converged and len(b.gauss_newton) <= 3
        np.testing.assert_allclose(b.rho_physical, a.rho_physical, rtol=1e-3,
                                   atol=1e-3 * abs(a.rho_physical).max())

    def test_scaling_auto(self, kernels):
        K, G, _ = kernels
        f, g = np.ones(K.shape[0]), 2 * np.ones(G.shape[0])
        (s1, s2), _ = gl.dataset_scales([f, g], "auto", linear=[True, True])
        assert s1 == 1.0 and s2 == 0.5
        (s1, s2), _ = gl.dataset_scales([f, g], "auto", [np.full(f.size, 2.0), 0.5],
                                        linear=[True, False])
        np.testing.assert_allclose(s1, 0.5)


@pytest.fixture(scope="module")
def dc_problem():
    """Gravity (linear) + DC resistivity (nonlinear) of one conductive, dense block."""
    from discretize import TensorMesh
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.dc_resistivity import DCResistivityMethod
    from geoinv3d.methods.gravity import GravityMethod
    h = [(50.0, 2, -1.5), (50.0, 12), (50.0, 2, 1.5)]
    tm = TensorMesh([h, h, [(50.0, 2, -1.5), (50.0, 8)]], origin="CCN")
    mesh = Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))
    cc = tm.cell_centers
    body = (abs(cc[:, 0]) < 100) & (abs(cc[:, 1]) < 100) & (cc[:, 2] < -75) & (cc[:, 2] > -225)
    rng = np.random.default_rng(0)
    xy = np.linspace(-250, 250, 9)
    X, Y = np.meshgrid(xy, xy)
    glocs = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 1.0)])
    G = np.asarray(GravityMethod().make_simulation_full(
        mesh, SurveyData(locations=glocs, observed=np.zeros(81), std=np.ones(81))).G)
    g = G @ (0.3 * body)
    sg = 0.02 * abs(g).max()
    g = g + rng.normal(scale=sg, size=g.size)
    rows = []
    for y in (-100.0, 0.0, 100.0):
        xs = np.arange(-275.0, 276.0, 50.0)
        for i in range(len(xs) - 1):
            for k in range(1, 5):
                j = i + 1 + k
                if j + 1 < len(xs):
                    rows.append([xs[i], y, 0, xs[i + 1], y, 0, xs[j], y, 0, xs[j + 1], y, 0])
    rows = np.array(rows)
    dc = DCResistivityMethod(sigma_background=1e-2)
    sim = dc.make_simulation_full(mesh, SurveyData(locations=rows, observed=np.zeros(len(rows)),
                                                   std=np.ones(len(rows))))
    m_true = np.full(tm.nC, np.log(1e-2))
    m_true[body] = np.log(1e-1)
    d = sim.dpred(m_true)
    sd = 0.03 * abs(d) + 1e-3 * abs(d).max()
    d = d + sd * rng.normal(size=d.size)
    return dict(tm=tm, mesh=mesh, body=body, G=G, g=g, sg=sg, sim=sim, d=d, sd=sd, rows=rows,
                glocs=glocs)


class TestNonlinear:
    def test_gravity_and_dc(self, dc_problem):
        p = dc_problem
        P = gl.GroupLassoProblem(
            [gl.GroupLassoData("gravity", p["g"], 0, p["sg"], operator=p["G"]),
             gl.GroupLassoData("dc", p["d"], 1, p["sd"], simulation=p["sim"])],
            model_names=["density", "log_conductivity"],
            references=[None, np.full(p["tm"].nC, np.log(1e-2))])
        assert all(np.ndim(s) for s in P.scales)          # "auto": std for nonlinear data
        seen = []
        r = P.solve(0.03 * P.lambda1_max(), 0.3, gn_callback=lambda k, rec: seen.append(rec))
        assert r.converged and P.gn_stop in ("kkt", "objective")
        objectives = [rec["objective"] for rec in seen]
        assert all(b <= a for a, b in zip(objectives, objectives[1:]))   # never worse
        # predictions are the forward modelling of the final model
        np.testing.assert_allclose(r.prediction("dc"), p["sim"].dpred(r.model("log_conductivity")),
                                   rtol=1e-10)
        chi2 = P.chi2_of(r)
        assert chi2["dc"] < 2 * len(p["d"]) and chi2["gravity"] < 3 * len(p["g"])
        body = p["body"]
        log_sigma = r.model("log_conductivity") / np.log(10)
        assert log_sigma[body].mean() > -1.6 and abs(log_sigma[~body].mean() + 2) < 0.05
        assert r.model("density")[body].mean() > 0.1

    def test_cross_gradient_with_dc(self, dc_problem):
        p = dc_problem
        make = lambda: gl.GroupLassoProblem(
            [gl.GroupLassoData("gravity", p["g"], 0, p["sg"], operator=p["G"]),
             gl.GroupLassoData("dc", p["d"], 1, p["sd"], simulation=p["sim"])],
            references=[None, np.full(p["tm"].nC, np.log(1e-2))], mesh=p["tm"])
        P = make()
        lam1 = 0.1 * P.lambda1_max()
        a = P.solve(lam1, 0.3)
        b = P.solve(lam1, 0.3, cross_gradient=0.1, state=a.state)
        assert b.final_terms["cross_gradient"] < P.cross_gradient_value(a.models_scaled)
        assert b.gauss_newton and b.cross_gradient == 0.1


class TestWorkerGeneral:
    def test_shared_density_model_and_dc(self, dc_problem):
        from geoinv3d.cloud.task import InversionTask
        from geoinv3d.cloud.worker import execute_task
        from geoinv3d.datamodel.survey import SurveyData
        from geoinv3d.methods.gravity import GravityMethod
        p = dc_problem
        tm, mesh = p["tm"], p["mesh"]
        gzz = GravityMethod("gzz").make_simulation_full(
            mesh, SurveyData(locations=p["glocs"], observed=np.zeros(81), std=np.ones(81))
        ).dpred(0.3 * p["body"])
        szz = 0.02 * abs(gzz).max()
        surveys = [{"locations": p["glocs"], "observed": p["g"], "std": np.full(81, p["sg"])},
                   {"locations": p["glocs"], "observed": gzz, "std": np.full(81, szz)},
                   {"locations": p["rows"], "observed": p["d"], "std": p["sd"]}]
        task = InversionTask(task_id="t", hx=tm.h[0], hy=tm.h[1], hz=tm.h[2],
                             origin=tuple(tm.origin), regularization_type="group_lasso",
                             joint_methods=["gravity", "gravity", "dc"],
                             joint_kwargs_list=[{}, {"component": "gzz"}, {}],
                             joint_models=["density", "density", None], joint_surveys=surveys,
                             gl_lambda1_selection="fixed", gl_lambda1_ratio=0.05)
        result = execute_task(task)
        assert list(result["recovered_models"]) == ["density", "dc_resistivity"]
        assert result["dataset_labels"] == ["gravity", "gravity_2", "dc_resistivity"]
        assert result["dataset_models"] == ["density", "density", "dc_resistivity"]
        assert set(result["joint_data"]) == {"gravity", "gravity_2", "dc_resistivity"}
        info = result["group_lasso"]
        assert info["models"] == {"density": ["gravity", "gravity_2"],
                                  "dc_resistivity": ["dc_resistivity"]}
        assert info["data_scales"]["gravity"] == "per datum"      # auto -> std
        assert info["gauss_newton"] and info["gauss_newton_stop"] in ("kkt", "objective",
                                                                      "max_iter")
        # one model only: nothing to couple
        task.joint_models = ["density", "density", "density"]
        with pytest.raises(ValueError, match="different properties|at least two"):
            execute_task(task)
        task.joint_methods, task.joint_models = ["gravity", "gravity"], ["density", "density"]
        task.joint_surveys, task.joint_kwargs_list = surveys[:2], [{}, {"component": "gzz"}]
        with pytest.raises(ValueError, match="at least two models"):
            execute_task(task)
