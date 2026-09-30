"""The coupling of a joint inversion as a choice of its own (geoinv3d/methods/coupling.py)."""

import contextlib
import io
import zipfile

import numpy as np
import pytest

from geoinv3d.methods.coupling import (
    COUPLINGS, check, coupling_label, linear_coefficients, pgi_settings, resolve,
)
from geoinv3d.methods.joint import JointInversion, MethodSetup, ModelRegularization
from tests.test_joint_regularization import BODY, MESH, surveys  # noqa: F401  (fixture)


def _run(setups, **kw):
    joint = JointInversion(setups, max_iter=25, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return joint, joint.run()


def _setups(surveys):
    return [MethodSetup(surveys[k][0], surveys[k][1], MESH, np.zeros(MESH.n_cells)) for k in ("gz", "tmi")]


def _corr(res):
    return np.corrcoef(res.recovered_models["gravity"], res.recovered_models["magnetics"])[0, 1]


class TestRegistry:
    def test_resolve_keeps_the_former_settings(self):
        assert resolve(None, "group_lasso") == "group_lasso"
        assert resolve(None, "sparse", 2.0) == "cross_gradient"
        assert resolve(None, "sparse", 0.0) == "none"
        assert resolve("JTV", "sparse") == "joint_total_variation"
        with pytest.raises(ValueError, match="Unknown coupling"):
            resolve("magic")
        with pytest.raises(ValueError, match="group lasso coupling"):
            resolve("cross_gradient", "group_lasso")

    def test_labels_families_and_limits(self):
        assert {c.family for c in COUPLINGS.values()} == {"structural", "petrophysical", "sparsity", "none"}
        assert "hybrid" in coupling_label("group_lasso+cross_gradient")
        check("cross_gradient", 3)
        with pytest.raises(ValueError, match="exactly 2"):
            check("linear_correspondence", 3)
        with pytest.raises(ValueError, match="at least two"):
            check("pgi", 1)

    def test_linear_relation(self):
        np.testing.assert_allclose(linear_coefficients({"slope": 6, "intercept": 0.1}), [1, -6, -0.1])
        np.testing.assert_allclose(linear_coefficients({"coefficients": [2, 1, 0]}), [2, 1, 0])
        with pytest.raises(ValueError, match="relation"):
            linear_coefficients({})

    def test_pgi_units(self):
        s = pgi_settings({"units": [{"name": "block", "means": {"gravity": 0.3, "magnetics": 0.05},
                                     "proportion": 0.05}]}, ["gravity", "magnetics"], [0.0, 0.0])
        bg, block = s.units
        assert bg.name == "background" and bg.means == [0.0, 0.0]
        np.testing.assert_allclose(bg.stds, [0.025 * 0.3, 0.025 * 0.05])       # tight, 2.5 %
        np.testing.assert_allclose(block.stds, [0.03, 0.005])                   # a tenth of the contrast
        assert np.isclose(bg.proportion + block.proportion, 1) and bg.proportion > 0.9
        with pytest.raises(ValueError, match="no means"):
            pgi_settings({"units": [{"name": "x", "means": {"gravity": 1}}]}, ["gravity", "magnetics"], [0, 0])


class TestGaussNewtonCouplings:
    def test_the_weight_is_unit_free(self, surveys):
        """Raw, the cross-gradient of g/cc and SI models is ~1e-10: weight 1 did nothing.
        Scaled against the regularization's curvature, weight 1 couples."""
        _, none = _run(_setups(surveys), regularization=ModelRegularization("l2"), coupling="none")
        joint, xg = _run(_setups(surveys), regularization=ModelRegularization("l2"),
                         coupling="cross_gradient", coupling_weight=1.0)
        _, raw = _run(_setups(surveys), regularization=ModelRegularization("l2"),
                      coupling="cross_gradient", coupling_weight=1.0, coupling_options={"scale": "raw"})
        assert _corr(xg) > _corr(none) + 0.03
        assert abs(_corr(raw) - _corr(none)) < 1e-3          # the former behaviour: no effect
        info = xg.extras["coupling"]
        assert info["kind"] == "cross_gradient" and info["scaling"] == "curvature"
        assert info["multipliers"][0] > 1e6                   # far above the raw 1
        for r in (none, xg):                                  # both fit both datasets
            assert all(20 < d["chi2"] < 100 for d in r.extras["datasets"].values())

    def test_joint_total_variation_is_balanced(self, surveys):
        """The models are scaled to comparable gradients: neither dataset is sacrificed."""
        for w in (1.0, 10.0):
            _, r = _run(_setups(surveys), regularization=ModelRegularization("l2"),
                        coupling="joint_total_variation", coupling_weight=w)
            assert all(20 < d["chi2"] < 100 for d in r.extras["datasets"].values())

    def test_linear_correspondence(self, surveys):
        _, r = _run(_setups(surveys), regularization=ModelRegularization("l2"),
                    coupling="linear_correspondence", coupling_options={"slope": 6.0})
        rho, chi = r.recovered_models["gravity"], r.recovered_models["magnetics"]
        assert _corr(r) > 0.99
        assert np.sqrt(np.mean((rho - 6 * chi) ** 2)) < 0.1 * np.abs(rho).max()

    def test_group_lasso_is_not_a_gauss_newton_coupling(self, surveys):
        with pytest.raises(ValueError, match="own solver"):
            JointInversion(_setups(surveys), coupling="group_lasso")


def test_pgi_recovers_and_classifies_the_block():
    """16 x 16 x 8 cells, 12 x 12 stations, a 4 x 4 x 3 block (0.3 g/cc, 0.05 SI)."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod

    mesh = Mesh3D.uniform(16, 16, 8, 50.0, 50.0, 50.0, origin=(0.0, 0.0, -400.0))
    cc = mesh.to_discretize().cell_centers
    body = ((cc[:, 0] > 300) & (cc[:, 0] < 500) & (cc[:, 1] > 300) & (cc[:, 1] < 500)
            & (cc[:, 2] > -250) & (cc[:, 2] < -100))
    xy = np.linspace(25, 775, 12)
    xx, yy = np.meshgrid(xy, xy)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 20.0)])
    rng = np.random.default_rng(0)
    setups = []
    for method, m in ((GravityMethod(), 0.3 * body),
                      (MagneticsMethod(inducing_field=(50000.0, 60.0, 10.0)), 0.05 * body)):
        s = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
        d = method.make_simulation_full(mesh, s).dpred(m.astype(float))
        sig = 0.02 * abs(d).max()
        survey = SurveyData(locations=locs, std=np.full(len(locs), sig),
                            observed=d + rng.normal(scale=sig, size=d.size), method=method.method_name)
        setups.append(MethodSetup(method, survey, mesh, np.zeros(mesh.n_cells)))
    units = [{"name": "block", "means": {"gravity": 0.3, "magnetics": 0.05}, "proportion": 0.05}]
    joint, r = _run(setups, coupling="pgi", coupling_options={"units": units})
    rho = r.recovered_models["gravity"]
    mem = r.extras["pgi"]["membership"]
    assert [u["name"] for u in r.extras["pgi"]["units"]] == ["background", "block"]
    assert np.sum((mem == 1) & body) >= 40 and np.sum((mem == 1) & ~body) < 40
    assert rho[body].mean() > 0.15          # an L2 inversion gives 0.045 here
    assert r.extras["models"]["gravity"]["regularization"] == "pgi"


class TestPipeline:
    def _files(self, tmp_path):
        from tests.test_data_pipeline import _station_grid, _synthetic, _write_csv
        locs = _station_grid()
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        _write_csv(tmp_path / "m.csv", locs, _synthetic("magnetics", locs))

    @pytest.mark.parametrize("coupling,options", [
        ("joint_total_variation", {}),
        ("pgi", {"units": [{"name": "block", "means": {"gravity": 0.3, "magnetics": 0.02}}]}),
    ])
    def test_a_coupling_through_the_pipeline_and_the_viewer(self, tmp_path, coupling, options):
        from tests.test_data_pipeline import _joint_params
        from geoinv3d.cloud.worker import pack_result, run_data_pipeline
        from geoinv3d.viz.result_workflow import build_workflow, load_result

        self._files(tmp_path)
        r = run_data_pipeline(_joint_params(["g.csv"], ["m.csv"], coupling=coupling,
                                            coupling_options=options, max_iter=6), str(tmp_path))
        assert r["coupling"]["kind"] == coupling and r["settings"]["coupling"] == coupling
        if coupling == "pgi":
            assert r["settings"]["regularization_type"] == "pgi" and "membership" in r["pgi"]
        path = pack_result(r, str(tmp_path / "result.zip"))
        run = load_result(path)
        run["_name"] = coupling
        other = dict(run, settings=dict(run["settings"], coupling="cross_gradient", coupling_weight=1.0),
                     _name="xg")
        wf = build_workflow([run, other])
        levels = [n["branch"]["level"] for n in wf["nodes"] if "branch" in n]
        assert "coupling" in levels
        names = {n["name"] for n in wf["nodes"] if n.get("branch", {}).get("level") == "coupling"}
        assert any(n.startswith(f"coupling: {coupling_label(coupling)}") for n in names)

    def test_former_parameters_still_mean_their_coupling(self, tmp_path):
        from geoinv3d.cloud.task import pack_task, unpack_task
        from geoinv3d.cloud.task import InversionTask

        t = InversionTask(task_id="t", method_type="joint", joint_methods=["gravity", "magnetic"],
                          regularization_type="group_lasso")
        assert t.coupling == "group_lasso"
        t = InversionTask(task_id="t", method_type="joint", joint_methods=["gravity", "magnetic"],
                          cross_gradient_weight=3.0)
        assert t.coupling == "cross_gradient" and t.effective_coupling_weight == 3.0
        t = InversionTask(task_id="t", method_type="joint", joint_methods=["gravity", "magnetic"],
                          joint_coupling="linear_correspondence", coupling_weight=0.5,
                          coupling_options={"slope": 6.0})
        path = pack_task(t, str(tmp_path / "task.zip"))
        u = unpack_task(path)
        assert (u.coupling, u.coupling_weight, u.coupling_options) == ("linear_correspondence", 0.5,
                                                                        {"slope": 6.0})
        with zipfile.ZipFile(path) as zf:
            assert b'"joint_coupling": "linear_correspondence"' in zf.read("meta.json")


def test_node_params_round_trip():
    from geoinv3d.nodes.inversion_nodes import JointRegularizedInversionNode
    n = JointRegularizedInversionNode(["gravity", "magnetic"], coupling="pgi",
                                      coupling_options={"units": [{"name": "a", "means": [0.1, 0.01]}]})
    m = JointRegularizedInversionNode.from_params(n.params(), None)
    assert (m.coupling, m.coupling_options) == ("pgi", n.coupling_options)
    old = JointRegularizedInversionNode(["gravity", "magnetic"], regularization_type="group_lasso")
    assert old.coupling == "group_lasso"
