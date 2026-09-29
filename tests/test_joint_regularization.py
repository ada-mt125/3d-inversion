"""Joint inversion (JointInversion) with any regularization per model."""

import contextlib
import io

import numpy as np
import pytest

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.gravity import GravityMethod
from geoinv3d.methods.joint import (
    JointInversion, MethodSetup, ModelRegularization, assign_models,
)
from geoinv3d.methods.magnetics import MagneticsMethod

FIELD = (50000.0, 60.0, 10.0)
MESH = Mesh3D.uniform(10, 10, 5, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -250.0))
CC = MESH.to_discretize().cell_centers
BODY = ((abs(CC[:, 0] - 250) < 80) & (abs(CC[:, 1] - 250) < 80)
        & (CC[:, 2] > -175) & (CC[:, 2] < -75))


def _locs():
    xy = np.linspace(25, 475, 7)
    xx, yy = np.meshgrid(xy, xy)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])


@pytest.fixture(scope="module")
def surveys():
    """Noisy gz, gzz and TMI data (2 % of the peak) of one block."""
    rng = np.random.default_rng(0)
    locs = _locs()
    out = {}
    for key, method, m in (("gz", GravityMethod(), 0.3 * BODY),
                           ("gzz", GravityMethod("gzz"), 0.3 * BODY),
                           ("tmi", MagneticsMethod(inducing_field=FIELD), 0.05 * BODY)):
        s = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
        d = method.make_simulation_full(MESH, s).dpred(m)
        sigma = 0.02 * abs(d).max()
        out[key] = (method, SurveyData(locations=locs, std=np.full(len(locs), sigma),
                                       observed=d + rng.normal(scale=sigma, size=d.size),
                                       method=method.method_name))
    return out


def _setups(surveys, keys=("gz", "tmi"), **kw):
    return [MethodSetup(surveys[k][0], surveys[k][1], MESH, np.zeros(MESH.n_cells), **kw)
            for k in keys]


def _run(joint):
    with contextlib.redirect_stdout(io.StringIO()):
        return joint.run()


def _single(method, survey, kind):
    from geoinv3d.cloud.task import InversionTask
    from geoinv3d.cloud.worker import run_single_inversion
    kwargs = {"inducing_field": list(FIELD)} if method.method_name == "magnetics" else {}
    task = InversionTask(task_id="s", method_type=method.method_name, method_kwargs=kwargs,
                         regularization_type=kind, l1l2_solver="irls", bounds_lower=0.0,
                         bounds_upper=1.0, max_iter=30, max_irls_iterations=10,
                         station_locations=survey.locations, observed_data=survey.observed,
                         data_std=survey.std, initial_model=np.zeros(MESH.n_cells))
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(run_single_inversion(task, mesh=MESH)["recovered_model"])


class TestModels:
    def test_labels(self, surveys):
        g, m = surveys["gz"], surveys["tmi"]
        setups = [MethodSetup(g[0], g[1], MESH, np.zeros(1)),
                  MethodSetup(g[0], g[1], MESH, np.zeros(1)),
                  MethodSetup(m[0], m[1], MESH, np.zeros(1), model="gravity_2")]
        models, datasets = assign_models(setups)
        # the explicit label is taken, so the second gravity dataset skips it
        assert models == ["gravity", "gravity_3", "gravity_2"]
        assert datasets == ["gravity", "gravity_2", "magnetics"]

    def test_shared_model_and_checks(self, surveys):
        setups = _setups(surveys, ("gz", "gzz", "tmi"))
        setups[0].model = setups[1].model = "density"
        joint = JointInversion(setups, regularization=ModelRegularization("l2"))
        assert [m.name for m in joint.models] == ["density", "magnetics"]
        assert joint.models[0].setups == [0, 1]
        assert joint.starting_model().shape == (2 * MESH.n_cells,)
        active = np.ones(MESH.n_cells, bool)
        active[:10] = False
        setups[1].active_cells = active
        with pytest.raises(ValueError, match="different meshes or active cells"):
            JointInversion(setups)

    def test_spec_validation(self):
        with pytest.raises(ValueError, match="kind"):
            ModelRegularization("l0")
        with pytest.raises(ValueError, match="depth_weighting"):
            ModelRegularization(depth_weighting="deep")
        with pytest.raises(ValueError, match="below"):
            ModelRegularization(lower=1.0, upper=0.0)
        r = ModelRegularization("sparse", norms=[0, 1, 1, 2], lower=0.0)
        assert ModelRegularization.from_dict(r.to_dict()) == r

    def test_legacy_path_is_unchanged(self, surveys):
        """Without a ModelRegularization the original WeightedLeastSquares path runs."""
        from simpeg import directives, regularization
        joint = JointInversion(_setups(surveys), reg_kwargs={"alpha_s": 1e-3})
        assert joint.legacy
        comps = joint.build()
        kinds = [type(d) for d in comps["inv"].directiveList.dList]
        assert directives.BetaSchedule in kinds
        assert all(type(r) is regularization.WeightedLeastSquares for r in comps["reg_list"])
        assert comps["balance"] is None


class TestRegularizations:
    @pytest.mark.parametrize("kind", ["l2", "sparse", "l1l2", "mgs", "tv"])
    def test_each_kind_matches_its_single_inversion(self, surveys, kind):
        """With a negligible coupling, the balanced joint inversion regularizes every
        model as its own inversion would, and fits both datasets to chi^2 ~ N."""
        reg = ModelRegularization(kind, lower=0.0, upper=1.0)
        joint = JointInversion(_setups(surveys), regularization=reg, max_iter=30,
                               max_irls_iterations=10)
        r = _run(joint)
        chi2 = {k: v["chi2"] for k, v in r.extras["datasets"].items()}
        assert all(30 < c < 80 for c in chi2.values()), chi2     # N = 49 each
        for name, key in (("gravity", "gz"), ("magnetics", "tmi")):
            single = _single(*surveys[key], kind)
            m = r.recovered_models[name]
            assert np.corrcoef(m, single)[0, 1] > 0.9, name
            assert m.min() >= 0.0 and m.max() <= 1.0
        assert {v["kind"] for v in r.extras["models"].values()} == {kind}
        assert all("balance" in v for v in r.extras["models"].values())

    def test_mixed_kinds_and_bounds(self, surveys):
        setups = _setups(surveys)
        setups[1].regularization = ModelRegularization("mgs", lower=0.0, upper=0.02)
        joint = JointInversion(setups, regularization=ModelRegularization("sparse"),
                               max_iter=20, max_irls_iterations=5, cross_gradient_weight=1.0)
        r = _run(joint)
        info = r.extras["models"]
        assert info["gravity"]["regularization"] == "sparse_IRLS"
        assert info["magnetics"]["regularization"] == "focusing_MGS"
        assert info["magnetics"]["focusing_threshold"] > 0
        chi = r.recovered_models["magnetics"]
        assert chi.min() >= 0.0 and chi.max() <= 0.02
        assert r.extras["datasets"]["gravity"]["predicted"].shape == (49,)

    def test_shared_density_model(self, surveys):
        """gz and gzz of one density model: one model, both datasets fitted."""
        setups = _setups(surveys, ("gz", "gzz", "tmi"))
        setups[0].model = setups[1].model = "density"
        r = _run(JointInversion(setups, regularization=ModelRegularization("l2"), max_iter=30))
        assert set(r.recovered_models) == {"density", "magnetics"}
        assert r.extras["models"]["density"]["datasets"] == ["gravity", "gravity_2"]
        chi2 = {k: v["chi2"] for k, v in r.extras["datasets"].items()}
        assert set(chi2) == {"gravity", "gravity_2", "magnetics"}
        assert all(c < 3 * 49 for c in chi2.values())
        assert r.recovered_models["density"][BODY].mean() > 3 * abs(
            r.recovered_models["density"][~BODY]).mean()


class TestDirectives:
    def test_sensitivity_weights_are_normalized_per_model(self, surveys):
        """SimPEG's global normalization would leave the less sensitive model's weights
        ~1e-3 at most; here each model's weights peak at 1."""
        from geoinv3d.methods.directives import JointSensitivityWeights
        joint = JointInversion(_setups(surveys), regularization=ModelRegularization("l2"))
        comps = joint.build()
        directive = next(d for d in comps["inv"].directiveList.dList
                         if isinstance(d, JointSensitivityWeights))
        comps["inv_prob"].model = joint.starting_model()
        directive.initialize()
        for reg in comps["reg_list"]:
            w = reg.objfcts[0].get_weights("sensitivity")
            assert np.isclose(w.max(), 1.0) and w.min() > 0

    def test_adaptive_balance_evens_the_misfits(self, surveys):
        """One beta alone can leave one dataset overfitted (the L1–L2 case below
        gave chi^2 = 103 and 1 before the adaptive step)."""
        chi2 = {}
        for adaptive in (False, True):
            joint = JointInversion(_setups(surveys), regularization=ModelRegularization("l1l2"),
                                   max_iter=30, max_irls_iterations=10)
            comps = joint.build()
            comps["balance"].adaptive = adaptive
            with contextlib.redirect_stdout(io.StringIO()):
                m = comps["inv"].run(joint.starting_model())
            chi2[adaptive] = []
            for dmis in comps["dmis_list"]:
                r = dmis.W @ (dmis.simulation.dpred(m) - dmis.data.dobs)
                chi2[adaptive].append(float(r @ r))
        spread = {k: max(v) / min(v) for k, v in chi2.items()}
        assert spread[True] < 1.5 < spread[False], chi2

    def test_preconditioner_applies_the_multipliers(self, surveys):
        from geoinv3d.methods.directives import JointUpdatePreconditioner
        joint = JointInversion(_setups(surveys), regularization=ModelRegularization("l2"),
                               cross_gradient_weight=3.0)
        comps = joint.build()
        inv_prob, reg = comps["inv_prob"], comps["combo_reg"]
        m = np.random.default_rng(0).normal(size=2 * MESH.n_cells) * 0.01
        inv_prob.model = m
        inv_prob.beta = 2.0
        reg.multipliers = [0.5, 4.0, 3.0]
        pc = next(d for d in comps["inv"].directiveList.dList
                  if isinstance(d, JointUpdatePreconditioner))
        pc.initialize()
        diag = sum(mult * r.deriv2(m).diagonal() for mult, r in zip(reg.multipliers, reg.objfcts))
        jtj = sum(s.getJtJdiag(m, W=d.W) for s, d in zip(comps["simulations"], comps["dmis_list"]))
        np.testing.assert_allclose(1.0 / comps["opt"].approxHinv.diagonal(), jtj + 2.0 * diag,
                                   rtol=1e-10)


def test_worker_overrides_and_legacy_types():
    from geoinv3d.cloud.task import InversionTask
    from geoinv3d.cloud.worker import joint_regularization
    task = InversionTask(task_id="t", regularization_type="sparse", alpha_s=None,
                         bounds_lower=-1.0, joint_methods=["gravity", "magnetics"],
                         joint_regularizations=[None, {"regularization_type": "tv",
                                                       "bounds_lower": 0.0, "alpha_x": 3.0}])
    g, m = joint_regularization(task, 0), joint_regularization(task, 1)
    assert (g.kind, g.lower, g.alpha_s) == ("sparse", -1.0, 1.0)
    assert (m.kind, m.lower, m.length_scale_x) == ("tv", 0.0, 3.0)
    task.regularization_type = "smooth"
    assert joint_regularization(task, 0) is None      # the legacy path
    task.joint_regularizations = [{"bogus": 1}, None]
    with pytest.raises(ValueError, match="bogus"):
        joint_regularization(task, 0)


def test_task_round_trip(tmp_path):
    from geoinv3d.cloud.task import InversionTask, pack_task, unpack_task
    task = InversionTask(task_id="t", joint_methods=["gravity", "gravity", "dc"],
                         joint_models=["density", "density", None],
                         joint_regularizations=[{"regularization_type": "mgs"}, None, None],
                         joint_balance=False, regularization_type="group_lasso",
                         gl_cross_gradient=0.1, gl_gn_max_iter=7)
    back = unpack_task(pack_task(task, str(tmp_path / "t.zip")))
    assert back.joint_models == ["density", "density", None]
    assert back.joint_regularizations == [{"regularization_type": "mgs"}, {}, {}]
    assert back.joint_balance is False
    assert (back.gl_cross_gradient, back.gl_gn_max_iter) == (0.1, 7)
