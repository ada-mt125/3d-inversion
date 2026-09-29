"""Depth weighting (Li & Oldenburg) as an alternative to sensitivity weighting."""

import numpy as np
import pytest

from geoinv3d.cloud.task import InversionTask, pack_task, unpack_task
from geoinv3d.cloud.worker import _depth_weights, _single_problem, run_data_pipeline
from geoinv3d.datamodel.mesh import Mesh3D
from tests.test_data_pipeline import _single, _station_grid, _synthetic, _write_csv


def _task(mesh=None, **kw):
    locs = _station_grid(0.0)
    if mesh is not None:
        kw["initial_model"] = np.zeros(mesh.to_discretize().n_cells)
    return InversionTask(task_id="t", method_type="gravity", station_locations=locs,
                         observed_data=np.zeros(len(locs)), data_std=np.ones(len(locs)),
                         regularization_type="sparse", **kw)


class TestWeights:
    def test_li_oldenburg_form(self):
        mesh = Mesh3D.uniform(4, 4, 5, 100.0, 100.0, 100.0, origin=(0.0, 0.0, -500.0))
        dmesh = mesh.to_discretize()
        for beta in (0.5, 1.0, 2.0):
            w = _depth_weights(_task(depth_weighting="depth", depth_weighting_exponent=beta), dmesh)
            depth = -dmesh.cell_centers[:, 2]            # stations at z = 0
            expected = (depth + 50.0) ** -beta           # z0 = half the smallest cell
            np.testing.assert_allclose(w, expected / expected.max())
        # SimPEG applies sqrt(weights): the norm weight is (z + z0)^(-beta/2)

    def test_set_on_the_regularization_instead_of_the_directive(self):
        mesh = Mesh3D.uniform(6, 6, 4, 100.0, 100.0, 100.0, origin=(0.0, 0.0, -400.0))
        task = _task(mesh, depth_weighting="depth", depth_weighting_exponent=1.0, max_iter=2)
        p = _single_problem(task, mesh)
        for obj in p.reg.objfcts:
            assert "depth" in obj.weights_keys and "sensitivity" not in obj.weights_keys
        default = _single_problem(_task(mesh, max_iter=2), mesh)
        assert all("depth" not in o.weights_keys for o in default.reg.objfcts)

    def test_unknown_method_and_negative_exponent(self):
        mesh = Mesh3D.uniform(4, 4, 3, 100.0, 100.0, 100.0, origin=(0.0, 0.0, -300.0))
        with pytest.raises(ValueError, match="depth_weighting"):
            _single_problem(_task(mesh, depth_weighting="distance"), mesh)
        with pytest.raises(ValueError, match="exponent"):
            _single_problem(_task(mesh, depth_weighting="depth", depth_weighting_exponent=-1),
                            mesh)

    def test_task_round_trip(self, tmp_path):
        task = _task(depth_weighting="depth", depth_weighting_exponent=1.5)
        back = unpack_task(pack_task(task, str(tmp_path / "t.zip")))
        assert back.depth_weighting == "depth" and back.depth_weighting_exponent == 1.5


class TestPipeline:
    @pytest.mark.parametrize("kind", ["sparse", "l2"])
    def test_depth_weighted_run(self, tmp_path, kind):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type=kind,
                         max_iter=8, alpha_s=1.0, depth_weighting="depth",
                         depth_weighting_exponent=1.0)
        result = run_data_pipeline(params, str(tmp_path))
        assert result["settings"]["depth_weighting"] == "depth"
        assert result["settings"]["depth_weighting_exponent"] == 1.0
        assert result["depth_weighting"] == "depth" and result["depth_weighting_exponent"] == 1.0
        assert result["n_iterations"] > 0 and np.isfinite(result["recovered_model"]).all()
        # the dense block: positive density dominates
        m = result["recovered_model"]
        assert m.max() > 0.02 and m.max() > abs(m.min())

    def test_depth_weighting_is_a_manual_setting(self, tmp_path):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        params = _single("gravity", ["g.csv"], param_mode="auto", max_iter=3,
                         depth_weighting="depth")
        result = run_data_pipeline(params, str(tmp_path))
        assert result["settings"]["depth_weighting"] == "sensitivity"
