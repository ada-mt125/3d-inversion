"""Tests for choosing beta: L-curve, GCV and fixed-beta IRLS."""

import numpy as np
import pytest
import scipy.sparse as sp

from geoinv3d.cloud.task import InversionTask, pack_task, unpack_task
from geoinv3d.cloud.worker import (
    _selection_point, run_fixed_beta, run_single_inversion,
)
from geoinv3d.methods.regparam import (
    gcv_minimum, gcv_score, influence_trace, lcurve_corner,
)


def _dense_trace(J, R, beta):
    R = R.toarray() if sp.issparse(R) else R
    return np.trace(J @ np.linalg.solve(J.T @ J + beta * R, J.T))


class TestLCurve:
    def test_symmetric_curve_has_corner_at_one(self):
        betas = np.logspace(-3, 3, 13)
        corner = lcurve_corner(betas, 1 + betas**2, 1 + betas**-2.0)
        assert corner == pytest.approx(1.0, rel=0.05)

    def test_order_of_points_does_not_matter(self):
        betas = np.logspace(-3, 3, 13)
        phi_d, phi_m = 1 + (betas / 10) ** 2, 1 + (betas / 10) ** -2.0
        perm = np.random.default_rng(0).permutation(len(betas))
        assert lcurve_corner(betas[perm], phi_d[perm], phi_m[perm]) == pytest.approx(
            lcurve_corner(betas, phi_d, phi_m))
        assert lcurve_corner(betas, phi_d, phi_m) == pytest.approx(10.0, rel=0.05)

    def test_coinciding_points_do_not_fake_a_corner(self):
        """A plateau (model pinned at a bound for large beta) is ignored."""
        betas = np.logspace(-3, 3, 13)
        phi_d, phi_m = 1 + betas**2, 1 + betas**-2.0
        phi_d[-3:] = phi_d[-3] * (1 + 1e-6 * np.arange(3))  # three near-identical points
        phi_m[-3:] = phi_m[-3] * (1 - 1e-6 * np.arange(3))
        assert lcurve_corner(betas, phi_d, phi_m) == pytest.approx(1.0, rel=0.1)

    def test_corner_validity(self):
        from geoinv3d.methods.regparam import lcurve_corner_info
        betas = np.logspace(-3, 3, 13)
        good = lcurve_corner_info(betas, 1 + betas**2, 1 + betas**-2.0)
        assert good["valid"] and good["curvature"] > 0
        t = np.log(betas)  # bends the wrong way everywhere: no corner
        bad = lcurve_corner_info(betas, np.exp(t), np.exp(-0.05 * t**2))
        assert bad["curvature"] <= 0 and not bad["valid"]

    def test_needs_four_points(self):
        with pytest.raises(ValueError, match="four"):
            lcurve_corner([1, 2, 3], [1, 2, 3], [3, 2, 1])


class TestGCV:
    @pytest.fixture
    def problem(self):
        rng = np.random.default_rng(1)
        J = rng.normal(size=(15, 40))
        lap = sp.diags([-1, 2.5, -1], [-1, 0, 1], shape=(40, 40))
        diag = sp.diags(rng.uniform(0.5, 2.0, 40))
        return J, lap.tocsr(), diag.tocsr()

    @pytest.mark.parametrize("beta", [1e-2, 1.0, 1e2])
    def test_trace_matches_dense(self, problem, beta):
        J, lap, diag = problem
        for R in (lap, diag):
            assert influence_trace(J, R, beta) == pytest.approx(_dense_trace(J, R, beta),
                                                                rel=1e-9)

    def test_trace_on_free_cells(self, problem):
        J, lap, _ = problem
        free = np.arange(40) % 3 != 0
        expected = _dense_trace(J[:, free], lap[free][:, free], 0.3)
        assert influence_trace(J, lap, 0.3, free=free) == pytest.approx(expected, rel=1e-9)
        assert influence_trace(J, lap, 0.3, free=np.zeros(40, bool)) == 0.0

    def test_trace_limits(self, problem):
        J, lap, _ = problem
        assert influence_trace(J, lap, 1e-10) == pytest.approx(J.shape[0], rel=1e-4)
        assert influence_trace(J, lap, 1e10) < 1e-6

    def test_gcv_picks_known_minimum(self):
        betas = np.logspace(-2, 2, 9)
        scores = 1 + (np.log(betas) - np.log(0.7)) ** 2  # minimum at 0.7
        assert gcv_minimum(betas, scores) == pytest.approx(0.7, rel=1e-6)
        assert gcv_score(4.0, 2.0, 10) == pytest.approx(10 * 4.0 / 64)


@pytest.fixture(scope="module")
def gravity_task():
    """12 x 12 x 6 cells under 64 stations; one dense block."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod

    mesh = Mesh3D.uniform(12, 12, 6, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -300.0))
    xy = np.linspace(25, 575, 8)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)])
    cc = mesh.to_discretize().cell_centers
    m_true = ((abs(cc[:, 0] - 300) < 80) & (abs(cc[:, 1] - 300) < 80)
              & (cc[:, 2] > -200) & (cc[:, 2] < -80)) * 0.3
    survey = SurveyData(locations=locs, observed=np.zeros(64), std=np.ones(64))
    d = GravityMethod().make_simulation_full(mesh, survey).dpred(m_true)
    sigma = 0.02 * abs(d).max()
    d = d + np.random.default_rng(3).normal(scale=sigma, size=d.size)

    def make(**kwargs):
        return InversionTask(
            task_id="beta", nx=12, ny=12, nz=6, dx=50.0, dy=50.0, dz=50.0,
            origin=(0.0, 0.0, -300.0), method_type="gravity",
            station_locations=locs, observed_data=d, data_std=np.full(64, sigma),
            initial_model=np.zeros(mesh.n_cells), max_iter=30, max_irls_iterations=15,
            **kwargs)

    return make


class TestFixedBeta:
    @pytest.mark.parametrize("reg_type", ["l1l2", "sparse", "l2"])
    def test_beta_stays_fixed(self, gravity_task, reg_type):
        task = gravity_task(regularization_type=reg_type, bounds_lower=0.0, bounds_upper=1.0)
        result, _, _ = run_fixed_beta(task, 0.37)
        assert [it["beta"] for it in result["iterations"]] == pytest.approx(
            [0.37] * result["n_iterations"])

    def test_larger_beta_fits_worse(self, gravity_task):
        task = gravity_task(regularization_type="l1l2", l1_ratio=0.5)
        phi_d = []
        for beta in (1e-2, 1.0, 1e2):
            result, p, inv_prob = run_fixed_beta(task, beta)
            phi_d.append(_selection_point(p, inv_prob, result["recovered_model"])["phi_d"])
        assert phi_d[0] < phi_d[1] < phi_d[2]

    def test_selection_point_gcv_matches_dense(self, gravity_task):
        task = gravity_task(regularization_type="l2", alpha_s=1e-2)
        result, p, inv_prob = run_fixed_beta(task, 5.0)
        m = result["recovered_model"]
        point = _selection_point(p, inv_prob, m)
        J = np.asarray(p.sim.G) / task.data_std[:, None]
        R = p.reg.deriv2(m) / 2
        assert point["trace_A"] == pytest.approx(_dense_trace(J, R, 5.0), rel=1e-6)
        r2 = np.sum(((p.sim.G @ m) - task.observed_data) ** 2 / task.data_std**2)
        assert point["phi_d"] == pytest.approx(r2, rel=1e-6)


class TestBetaSelection:
    @pytest.mark.parametrize("criterion", ["lcurve", "gcv"])
    def test_selection_through_worker(self, gravity_task, criterion):
        factors = tuple(np.logspace(-2, 2, 7))
        task = gravity_task(regularization_type="l1l2", l1l2_solver="irls", l1_ratio=0.5,
                            bounds_lower=0.0, bounds_upper=1.0, beta_selection=criterion,
                            beta_sweep_factors=factors)
        result = run_single_inversion(task)
        sel = result["beta_selection"]
        assert sel["criterion"] == criterion
        assert len(sel["beta"]) == 7 and all(g is not None for g in sel["gcv"])
        assert min(sel["beta"]) <= sel["beta_chosen"] <= max(sel["beta"])
        assert sel["beta_chosen"] == sel[f"beta_{criterion}"]
        assert result["iterations"][-1]["beta"] == pytest.approx(sel["beta_chosen"])
        # phi_d grows with beta along the sweep (betas are stored largest first)
        assert np.all(np.diff(sel["phi_d"]) <= 1e-6 * max(sel["phi_d"]))
        assert result["discrepancy_model"].shape == result["recovered_model"].shape

    def test_unknown_criterion(self, gravity_task):
        with pytest.raises(ValueError, match="beta_selection"):
            run_single_inversion(gravity_task(beta_selection="aic"))

    def test_task_roundtrip(self, gravity_task, tmp_path):
        task = gravity_task(beta_selection="gcv", beta_sweep=[1.0, 2.0, 4.0, 8.0],
                            beta_sweep_factors=(0.1, 1.0, 10.0))
        path = str(tmp_path / "t.zip")
        pack_task(task, path)
        task2 = unpack_task(path)
        assert task2.beta_selection == "gcv"
        assert task2.beta_sweep == [1.0, 2.0, 4.0, 8.0]
        assert task2.beta_sweep_factors == (0.1, 1.0, 10.0)
