"""Tests for the MGS / TV focusing regularization."""

import discretize
import numpy as np
import pytest
from simpeg.regularization import WeightedLeastSquares

from geoinv3d.methods.regularization import Focusing


@pytest.fixture(scope="module")
def mesh():
    return discretize.TensorMesh([np.full(6, 10.0), np.full(5, 20.0), np.full(4, 15.0)])


def _reg(mesh, stabilizer):
    reg = Focusing(mesh, stabilizer=stabilizer, alpha_s=0.0,
                   reference_model=np.zeros(mesh.n_cells))
    return reg


@pytest.mark.parametrize("stabilizer", ["mgs", "tv"])
class TestMajorizer:
    def test_gradient_matches_stabilizer(self, mesh, stabilizer):
        reg = _reg(mesh, stabilizer)
        mk = np.random.default_rng(0).normal(size=mesh.n_cells)
        reg.update_weights(mk)  # sets e and the weights at mk
        h = 1e-6
        numeric = np.array([
            (reg.stabilizer_value(mk + h * e) - reg.stabilizer_value(mk - h * e)) / (2 * h)
            for e in np.eye(mesh.n_cells)
        ])
        np.testing.assert_allclose(reg.deriv(mk), numeric, rtol=1e-5,
                                   atol=1e-8 * abs(numeric).max())

    def test_quadratic_majorizes_stabilizer(self, mesh, stabilizer):
        reg = _reg(mesh, stabilizer)
        rng = np.random.default_rng(1)
        mk = rng.normal(size=mesh.n_cells)
        reg.update_weights(mk)
        s_k, q_k = reg.stabilizer_value(mk), reg(mk)
        for _ in range(20):
            m = mk + rng.normal(scale=rng.uniform(0.1, 3.0), size=mesh.n_cells)
            assert reg(m) - q_k >= reg.stabilizer_value(m) - s_k - 1e-9 * abs(q_k)

    def test_l2_stage_is_plain_smoothness(self, mesh, stabilizer):
        reg = _reg(mesh, stabilizer)
        reg.norms = [2.0, 2.0, 2.0, 2.0]  # what UpdateIRLS does before IRLS starts
        m = np.random.default_rng(2).normal(size=mesh.n_cells)
        reg.update_weights(m)
        plain = WeightedLeastSquares(mesh, alpha_s=0.0, reference_model=np.zeros(mesh.n_cells))
        assert reg(m) == pytest.approx(plain(m))
        assert reg.focusing_threshold is None


def test_weights_and_threshold(mesh):
    reg = _reg(mesh, "mgs")
    m = np.zeros(mesh.n_cells)
    m[:30] = 1.0  # one sharp edge
    reg.update_weights(m)
    s = reg.gradient_magnitude2(m)
    e = reg.focusing_threshold
    assert e == pytest.approx(np.percentile(np.sqrt(s), 95.0))
    assert _reg(mesh, "tv").threshold_scale == 0.1
    w = reg.cell_weights(s, e**2)
    assert w.max() == pytest.approx(1.0) and w.min() < 0.3  # edges are released
    tv = _reg(mesh, "tv")
    tv.update_weights(m)
    w_tv = tv.cell_weights(s, e**2)
    assert np.all(w_tv >= w - 1e-12)  # TV releases edges less than MGS
    with pytest.raises(ValueError, match="stabilizer"):
        Focusing(mesh, stabilizer="l0")


@pytest.fixture(scope="module")
def gravity_task():
    """12 x 12 x 6 cells under 64 stations; one dense block."""
    from geoinv3d.cloud.task import InversionTask
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod

    mesh3d = Mesh3D.uniform(12, 12, 6, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -300.0))
    xy = np.linspace(25, 575, 8)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 10.0)])
    cc = mesh3d.to_discretize().cell_centers
    m_true = ((abs(cc[:, 0] - 300) < 80) & (abs(cc[:, 1] - 300) < 80)
              & (cc[:, 2] > -200) & (cc[:, 2] < -80)) * 0.3
    survey = SurveyData(locations=locs, observed=np.zeros(64), std=np.ones(64))
    d = GravityMethod().make_simulation_full(mesh3d, survey).dpred(m_true)
    sigma = 0.02 * abs(d).max()
    d = d + np.random.default_rng(3).normal(scale=sigma, size=d.size)

    def make(**kwargs):
        return InversionTask(
            task_id="focus", nx=12, ny=12, nz=6, dx=50.0, dy=50.0, dz=50.0,
            origin=(0.0, 0.0, -300.0), method_type="gravity",
            station_locations=locs, observed_data=d, data_std=np.full(64, sigma),
            initial_model=np.zeros(mesh3d.n_cells), max_iter=40, max_irls_iterations=20,
            alpha_s=1e-3, **kwargs)

    return make, mesh3d


def _gradient_support(m, mesh3d):
    """Fraction of cells whose gradient magnitude exceeds 10 % of its maximum."""
    reg = Focusing(mesh3d.to_discretize(), alpha_s=0.0)
    g = np.sqrt(reg.gradient_magnitude2(m))
    return float(np.mean(g > 0.1 * g.max()))


class TestWorker:
    @pytest.mark.parametrize("kind, label", [("mgs", "focusing_MGS"), ("tv", "total_variation")])
    def test_focusing_through_worker(self, gravity_task, kind, label):
        from geoinv3d.cloud.worker import run_single_inversion

        make, mesh3d = gravity_task
        result = run_single_inversion(make(regularization_type=kind))
        # L2 with the same (sensitivity) weighting: sparse with all norms 2
        smooth = run_single_inversion(make(regularization_type="sparse",
                                           norms=(2.0, 2.0, 2.0, 2.0)))
        assert result["regularization"] == label
        assert result["focusing_threshold"] > 0
        assert abs(result["iterations"][-1]["phi_d"] / 64 - 1) < 0.25
        m = result["recovered_model"]
        # focusing concentrates the gradients: smaller gradient support than L2
        assert _gradient_support(m, mesh3d) < _gradient_support(
            smooth["recovered_model"], mesh3d)

    def test_beta_sweep_keeps_focusing_parameter(self, gravity_task):
        from geoinv3d.cloud.worker import run_single_inversion

        make, _ = gravity_task
        task = make(regularization_type="mgs", beta_selection="lcurve",
                    beta_sweep_factors=tuple(np.logspace(-1, 1, 5)))
        result = run_single_inversion(task)
        sel = result["beta_selection"]
        assert np.all(np.diff(sel["phi_d"]) <= 1e-6 * max(sel["phi_d"]))
        assert result["focusing_threshold"] > 0
