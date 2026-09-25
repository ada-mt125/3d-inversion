"""Tests for the Utsugi (2019) L1–L2 coordinate-descent inversion."""

import numpy as np
import pytest
from scipy.optimize import minimize

from geoinv3d.cloud.task import InversionTask
from geoinv3d.cloud.worker import run_single_inversion
from geoinv3d.methods.l1l2_cda import (
    column_scaling, coordinate_descent, elastic_net_dof, elastic_net_path, invert_l1l2,
    lambda_max, penalty,
)


@pytest.fixture(scope="module")
def correlated():
    """Under-determined problem with correlated columns, like potential fields."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 8))
    X = np.repeat(base, 5, axis=1) + 0.3 * rng.normal(size=(20, 40))
    b_true = np.zeros(40)
    b_true[[3, 4, 17]] = [2.0, 1.5, -1.0]
    f = X @ b_true + 0.1 * rng.normal(size=20)
    return X, f


def _objective(X, f, lam, alpha):
    def value(b):
        r = f - X @ b
        return 0.5 * r @ r + lam * penalty(b, alpha)
    return value


def _reference(X, f, lam, alpha, lower=None, upper=None):
    """Independent solver: b = p - q with p, q >= 0 (smooth, bound constrained)."""
    m = X.shape[1]
    lo = -np.inf if lower is None else lower
    hi = np.inf if upper is None else upper

    def fun(z):
        p, q = z[:m], z[m:]
        b = p - q
        r = f - X @ b
        val = 0.5 * r @ r + lam * (0.5 * (1 - alpha) * b @ b + alpha * (p.sum() + q.sum()))
        g = -X.T @ r + lam * (1 - alpha) * b
        return val, np.concatenate([g + lam * alpha, -g + lam * alpha])

    bounds = [(max(0, lo), max(0, hi))] * m + [(max(0, -hi), max(0, -lo))] * m
    z = minimize(fun, np.zeros(2 * m), jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 50000}).x
    return z[:m] - z[m:]


class TestCoordinateDescent:
    @pytest.mark.parametrize("alpha", [1.0, 0.9, 0.5, 0.0])
    def test_matches_independent_solver(self, correlated, alpha):
        X, f = correlated
        lam = 0.2 * lambda_max(X, f, max(alpha, 0.5))
        b, _ = coordinate_descent(X, f, lam, alpha, tol=1e-12)
        ref = _reference(X, f, lam, alpha)
        obj = _objective(X, f, lam, alpha)
        assert obj(b) <= obj(ref) + 1e-9 * abs(obj(ref))
        np.testing.assert_allclose(b, ref, atol=1e-5 * abs(ref).max())

    @pytest.mark.parametrize("alpha", [1.0, 0.9, 0.5])
    def test_kkt_conditions(self, correlated, alpha):
        X, f = correlated
        lam = 0.1 * lambda_max(X, f, alpha)
        b, _ = coordinate_descent(X, f, lam, alpha, tol=1e-13)
        g = X.T @ (f - X @ b) - lam * (1 - alpha) * b  # = lam a sign(b) at optimum
        nz = b != 0
        np.testing.assert_allclose(g[nz], lam * alpha * np.sign(b[nz]), atol=1e-8 * lam)
        assert np.all(abs(g[~nz]) <= lam * alpha * (1 + 1e-8))
        if alpha == 1.0:
            assert nz.sum() <= X.shape[0]  # L1: at most N non-zeros (the paper's drawback)

    def test_bounds(self, correlated):
        X, f = correlated
        lam = 0.05 * lambda_max(X, f, 0.9)
        b, _ = coordinate_descent(X, f, lam, 0.9, lower=0.0, upper=1.2, tol=1e-12)
        assert b.min() >= 0.0 and b.max() <= 1.2
        ref = _reference(X, f, lam, 0.9, lower=0.0, upper=1.2)
        obj = _objective(X, f, lam, 0.9)
        assert obj(b) <= obj(ref) + 1e-9 * abs(obj(ref))

    @pytest.mark.parametrize("alpha, bounds", [(0.99, (None, None)), (0.9, (0.0, 1.2)),
                                               (0.5, (None, None))])
    def test_polishing_gives_same_minimizer(self, correlated, alpha, bounds):
        X, f = correlated
        lam = 0.02 * lambda_max(X, f, alpha)
        plain, n_plain = coordinate_descent(X, f, lam, alpha, lower=bounds[0],
                                            upper=bounds[1], tol=1e-12, polish_every=0)
        fast, n_fast = coordinate_descent(X, f, lam, alpha, lower=bounds[0],
                                          upper=bounds[1], tol=1e-12, polish_every=5)
        np.testing.assert_allclose(fast, plain, atol=1e-7 * abs(plain).max())
        assert n_fast <= n_plain

    def test_single_variable_soft_threshold(self):
        """The paper's Eq. (26) for one column: S(x^T f, lam a) / (x^T x + lam(1-a))."""
        x = np.array([[1.0], [2.0], [-0.5]])
        f = np.array([3.0, 1.0, 2.0])
        for lam, alpha in [(0.5, 1.0), (2.0, 0.6), (10.0, 0.9)]:
            xtf, xtx = float(x[:, 0] @ f), float(x[:, 0] @ x[:, 0])
            expected = np.sign(xtf) * max(abs(xtf) - lam * alpha, 0) / (xtx + lam * (1 - alpha))
            b, _ = coordinate_descent(x, f, lam, alpha)
            assert b[0] == pytest.approx(expected)

    def test_lambda_max_gives_zero_model(self, correlated):
        X, f = correlated
        top = lambda_max(X, f, 0.7)
        assert not np.any(coordinate_descent(X, f, top * 1.0001, 0.7)[0])
        assert np.any(coordinate_descent(X, f, top * 0.99, 0.7)[0])


class TestPath:
    def test_warm_start_equals_cold_start(self, correlated):
        X, f = correlated
        path = elastic_net_path(X, f, 0.9, n_decades=2, step=0.25, tol=1e-12)
        assert np.all(np.diff(path.lambdas) < 0)
        for lam, b in zip(path.lambdas[::3], path.betas[::3]):
            cold, _ = coordinate_descent(X, f, lam, 0.9, tol=1e-12)
            np.testing.assert_allclose(b, cold, atol=1e-7 * max(abs(cold).max(), 1e-12))
        # misfit grows and the penalty shrinks with lambda
        assert np.all(np.diff(path.residual_norm) <= 1e-9)
        assert np.all(np.diff(path.penalty) >= -1e-9)

    def test_elastic_net_dof(self, correlated):
        X, f = correlated
        lam, alpha = 0.05 * lambda_max(X, f, 0.8), 0.8
        b, _ = coordinate_descent(X, f, lam, alpha, tol=1e-12)
        A = b != 0
        XA = X[:, A]
        dense = np.trace(XA @ np.linalg.solve(XA.T @ XA + lam * (1 - alpha) * np.eye(A.sum()),
                                              XA.T))
        assert elastic_net_dof(X, b, lam, alpha) == pytest.approx(dense, rel=1e-8)
        b1, _ = coordinate_descent(X, f, lam, 1.0, tol=1e-12)
        assert elastic_net_dof(X, b1, lam, 1.0) == np.linalg.matrix_rank(X[:, b1 != 0])


class TestWeighting:
    def test_column_scaling(self):
        K = np.array([[3.0, 0.0], [4.0, 2.0]])
        np.testing.assert_allclose(column_scaling(K, "S2"), [5.0, 2.0])
        np.testing.assert_allclose(column_scaling(K, "S1"), np.sqrt([5.0, 2.0]))
        np.testing.assert_allclose(np.linalg.norm(K / column_scaling(K, "S2"), axis=0), 1.0)
        with pytest.raises(ValueError, match="weighting"):
            column_scaling(K, "S3")

    def test_s2_does_not_depend_on_model_units(self, correlated):
        X, f = correlated
        G = abs(X) * np.linspace(1.0, 0.05, X.shape[1])  # decaying sensitivity
        d = G @ np.where(np.arange(40) % 9 == 0, 0.02, 0.0)
        runs = {(w, u): invert_l1l2(G, d, 0.9, weighting=w, model_unit=u, n_decades=3)[0]
                for w in ("S1", "S2") for u in (1.0, 33.4)}
        np.testing.assert_allclose(runs["S2", 1.0].model, runs["S2", 33.4].model,
                                   rtol=1e-6, atol=1e-10)
        assert not np.allclose(runs["S1", 1.0].model, runs["S1", 33.4].model, rtol=1e-3)


@pytest.fixture(scope="module")
def magnetic_task():
    """12 x 12 x 6 cells, 64 TMI stations, one susceptible block."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.magnetics import MagneticsMethod

    field = (50000.0, 60.0, 10.0)
    mesh = Mesh3D.uniform(12, 12, 6, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -300.0))
    xy = np.linspace(25, 575, 8)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    cc = mesh.to_discretize().cell_centers
    m_true = ((abs(cc[:, 0] - 300) < 80) & (abs(cc[:, 1] - 300) < 80)
              & (cc[:, 2] > -200) & (cc[:, 2] < -80)) * 0.05
    survey = SurveyData(locations=locs, observed=np.zeros(64), std=np.ones(64))
    sim = MagneticsMethod(inducing_field=field).make_simulation_full(mesh, survey)
    d = sim.dpred(m_true)
    sigma = 0.02 * abs(d).max()
    d = d + np.random.default_rng(3).normal(scale=sigma, size=d.size)

    def make(**kwargs):
        return InversionTask(
            task_id="cda", nx=12, ny=12, nz=6, dx=50.0, dy=50.0, dz=50.0,
            origin=(0.0, 0.0, -300.0), method_type="magnetics",
            method_kwargs={"inducing_field": field}, regularization_type="l1l2",
            station_locations=locs, observed_data=d, data_std=np.full(64, sigma),
            initial_model=np.zeros(mesh.n_cells), lambda_decades=3.0, **kwargs)

    return make, np.asarray(sim.G), m_true


class TestWorker:
    @pytest.mark.parametrize("selection", ["auto", "discrepancy", "gcv"])
    def test_cda_through_worker(self, magnetic_task, selection):
        make, G, m_true = magnetic_task
        task = make(l1_ratio=0.9, beta_selection=selection)
        result = run_single_inversion(task)
        info = result["l1l2"]
        assert result["regularization"] == "elastic_net_CDA"
        if selection == "auto":  # L-curve, or chi^2 = N when the curve has no corner
            assert info["criterion"] in ("lcurve", "discrepancy")
        else:
            assert info["criterion"] == selection
        assert info["weighting"] == "S2"
        lams = np.asarray(info["lambdas"])
        assert lams.min() < info["lambda_opt"] < lams.max()
        assert result["n_iterations"] == len(lams) == 31
        m = result["recovered_model"]
        chi2 = np.sum(((G @ m - task.observed_data) / task.data_std) ** 2)
        assert info["chi2_opt"] == pytest.approx(chi2)
        if selection == "discrepancy":
            assert chi2 == pytest.approx(64, rel=0.05)
        # Susceptibility units: the peak is at a sane value, within a cell of the
        # block (x, y in 220-380 m, z in -200 to -80 m; cells are 50 m)
        assert 0.01 < m.max() < 0.5
        from geoinv3d.datamodel.mesh import Mesh3D
        cc = Mesh3D.uniform(12, 12, 6, 50.0, 50.0, 50.0,
                            origin=(0.0, 0.0, -300.0)).to_discretize().cell_centers
        x, y, z = cc[np.argmax(m)]
        assert abs(x - 300) < 130 and abs(y - 300) < 130 and -250 < z < -30

    def test_irls_still_available_and_unknown_solver(self, magnetic_task):
        make, _, _ = magnetic_task
        result = run_single_inversion(make(l1l2_solver="irls", max_iter=20,
                                           max_irls_iterations=5))
        assert result["regularization"] == "elastic_net_IRLS"
        with pytest.raises(ValueError, match="l1l2_solver"):
            run_single_inversion(make(l1l2_solver="admm"))
