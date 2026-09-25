"""Tests for the L1–L2 (elastic net) regularization and its directives."""

from types import SimpleNamespace

import numpy as np
import pytest
from simpeg import data_misfit, directives, inverse_problem, inversion, optimization
from simpeg.directives._regularization import IRLSMetrics
from simpeg.regularization import Smallness

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.directives import DampedUpdateIRLS, ElasticNetSensitivityWeights
from geoinv3d.methods.gravity import GravityMethod
from geoinv3d.methods.regularization import ElasticNet


@pytest.fixture(scope="module")
def small_problem():
    """A 32-cell gravity problem with 49 stations (well conditioned)."""
    mesh = Mesh3D.uniform(4, 4, 2, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -100.0))
    xx, yy = np.meshgrid(np.linspace(0, 200, 7), np.linspace(0, 200, 7))
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 5.0)])
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = GravityMethod().make_simulation_full(mesh, survey)
    dmesh = mesh.to_discretize()
    cc = dmesh.cell_centers
    m_true = ((abs(cc[:, 0] - 100) < 30) & (abs(cc[:, 1] - 100) < 30) & (cc[:, 2] > -60)) * 0.5
    sigma = 0.002
    d = sim.G @ m_true + np.random.default_rng(1).normal(scale=sigma, size=len(locs))
    return SimpleNamespace(dmesh=dmesh, sim=sim, G=np.asarray(sim.G), d=d, sigma=sigma,
                           n=dmesh.n_cells)


def _elastic_net(problem, l1_ratio):
    reg = ElasticNet(problem.dmesh, l1_ratio=l1_ratio, reference_model=np.zeros(problem.n))
    column_norms = np.sqrt(problem.sim.getJtJdiag(np.zeros(problem.n)))
    reg.objfcts[0].set_weights(sensitivity=column_norms)
    return reg


def _coordinate_descent(AtA, Atb, lam1, lam2, tol=1e-13, max_sweeps=50000):
    """Exact minimiser of ||Am - b||^2 + sum(lam1 |m| + lam2 m^2) (glmnet-style)."""
    m = np.zeros(len(Atb))
    diag = np.diag(AtA)
    for _ in range(max_sweeps):
        m_old = m.copy()
        for j in range(len(m)):
            rho = Atb[j] - AtA[j] @ m + diag[j] * m[j]
            m[j] = np.sign(rho) * max(abs(rho) - lam1[j] / 2, 0.0) / (diag[j] + lam2[j])
        if np.max(abs(m - m_old)) < tol:
            return m
    raise AssertionError("coordinate descent did not converge")


class TestElasticNetObjective:
    @pytest.mark.parametrize("l1_ratio", [1.0, 0.5, 0.0])
    def test_irls_reaches_exact_elastic_net_minimum(self, small_problem, l1_ratio):
        """Majorize-minimize with our weights converges to the exact elastic net."""
        p = small_problem
        reg = _elastic_net(p, l1_ratio)
        term = reg.objfcts[0]
        term.irls_threshold = 1e-9
        A, b = p.G / p.sigma, p.d / p.sigma
        AtA, Atb = A.T @ A, A.T @ b
        beta = 1e-3

        # Exact solution of ||Am - b||^2 + beta * phi_m (SimPEG scaling)
        s, v = term.depth_scaling(), p.dmesh.cell_volumes
        lam1 = beta * 2 * l1_ratio * v * s
        lam2 = beta * (1 - l1_ratio) * v * s**2
        if l1_ratio == 0.0:
            m_exact = np.linalg.solve(AtA + np.diag(lam2), Atb)
        else:
            m_exact = _coordinate_descent(AtA, Atb, lam1, lam2)

        m = np.linalg.solve(AtA + beta * np.diag(v * s**2), Atb)
        for _ in range(2000):
            reg.update_weights(m)
            m_new = np.linalg.solve(AtA + beta * reg.deriv2(m).toarray() / 2, Atb)
            if np.max(abs(m_new - m)) < 1e-14:
                break
            m = m_new

        np.testing.assert_allclose(m_new, m_exact, atol=1e-6 * abs(m_exact).max())
        np.testing.assert_array_equal(abs(m_new) > 1e-6, m_exact != 0)
        if l1_ratio == 1.0:
            assert np.sum(m_exact == 0) > p.n // 2  # L1 really is sparse

    def test_gradient_matches_elastic_net_value(self, small_problem):
        reg = _elastic_net(small_problem, 0.5)
        reg.objfcts[0].irls_threshold = 1e-3
        m = np.random.default_rng(2).normal(size=small_problem.n)
        reg.update_weights(m)  # majorizer touches the objective at m
        h = 1e-6
        numeric = np.array([
            (reg.elastic_net_value(m + h * e) - reg.elastic_net_value(m - h * e)) / (2 * h)
            for e in np.eye(small_problem.n)
        ])
        np.testing.assert_allclose(reg.deriv(m), numeric, rtol=1e-6)

    def test_l2_warmup_stage_is_plain_smallness(self, small_problem):
        reg = _elastic_net(small_problem, 0.8)
        reg.norms = [2.0]  # what UpdateIRLS does before switching to IRLS
        m = np.random.default_rng(3).normal(size=small_problem.n)
        reg.update_weights(m)
        plain = Smallness(small_problem.dmesh, reference_model=np.zeros(small_problem.n))
        plain.set_weights(sensitivity=reg.objfcts[0].get_weights("sensitivity"))
        assert np.isclose(reg(m), plain(m))

    def test_parameter_validation(self, small_problem):
        with pytest.raises(ValueError, match="l1_ratio"):
            ElasticNet(small_problem.dmesh, l1_ratio=1.5)
        reg = ElasticNet(small_problem.dmesh)
        with pytest.raises(ValueError, match="norm must be 1"):
            reg.norms = [0.5]

    def test_active_cells(self, small_problem):
        active = small_problem.dmesh.cell_centers[:, 2] < -50
        reg = ElasticNet(small_problem.dmesh, l1_ratio=0.5, active_cells=active)
        assert reg.nP == int(active.sum())


class TestDirectives:
    def test_sensitivity_weights_are_column_norms(self, small_problem):
        p = small_problem
        survey = p.sim.survey
        from simpeg import data as simpeg_data
        dobs = simpeg_data.Data(survey, dobs=p.d, standard_deviation=np.full(len(p.d), p.sigma))
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=p.sim)
        reg = ElasticNet(p.dmesh, reference_model=np.zeros(p.n))
        inv_prob = inverse_problem.BaseInvProblem(
            dmis, reg, optimization.InexactGaussNewton(maxIter=1))
        directive = ElasticNetSensitivityWeights()
        inversion.BaseInversion(inv_prob, directiveList=[directive])
        inv_prob.model = np.zeros(p.n)
        directive.initialize()
        np.testing.assert_allclose(reg.objfcts[0].get_weights("sensitivity"),
                                   np.linalg.norm(p.G, axis=0), rtol=1e-6)  # G is float32

    @staticmethod
    def _steer(directive_cls, n_iter=30):
        """Drive the IRLS beta control against a steep phi_d(beta) curve."""
        target = 169.0
        directive = directive_cls()
        directive._metrics = IRLSMetrics(input_norms=[])
        directive.metrics.start_irls_iter = 0
        directive.misfit_from_chi_factor = lambda chifact: target
        state = SimpleNamespace(phi_d=0.0)
        directive.inversion = SimpleNamespace(invProb=state)
        beta = 0.7  # phi_d grows steeply with beta and hits the target at beta = 1
        history = []
        for _ in range(n_iter):
            state.phi_d = target * beta ** 2.2
            history.append(state.phi_d)
            directive.adjust_cooling_schedule()
            beta /= directive.cooling_factor
        return np.array(history) / target

    def test_damped_irls_converges_where_plain_cycles(self):
        plain = self._steer(directives.UpdateIRLS)
        damped = self._steer(DampedUpdateIRLS)
        # SimPEG's controller settles into a cycle that misses the tolerance...
        assert np.any(abs(plain[-6:] - 1) > 0.1)
        # ...the damped one brackets the target and stays within it
        assert np.all(abs(damped[-6:] - 1) <= 0.1)
