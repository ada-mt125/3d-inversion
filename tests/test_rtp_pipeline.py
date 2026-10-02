"""Inverting magnetic data reduced to the pole: the reduction against a vertical-field
forward model, and a job that runs on it."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")
from geoinv3d.cloud import worker   # noqa: E402
from geoinv3d.datamodel.mesh import Mesh3D   # noqa: E402
from geoinv3d.datamodel.survey import SurveyData   # noqa: E402
from geoinv3d.methods.magnetics import MagneticsMethod   # noqa: E402
from tests.test_data_pipeline import SMALL_MESH, _write_csv   # noqa: E402

FIELD = (50000.0, 60.0, 10.0)


def _stations(step=50.0):
    xs = np.arange(-1500, 2101, step)
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def _tmi(locs, field):
    """A block (200-400 m in x/y, 100-300 m deep), 0.02 SI, under ``field``."""
    mesh = Mesh3D.uniform(10, 10, 6, 100.0, 100.0, 100.0, origin=(-200.0, -200.0, -600.0))
    cc = mesh.to_discretize().cell_centers
    block = ((cc[:, 0] > 200) & (cc[:, 0] < 400) & (cc[:, 1] > 200) & (cc[:, 1] < 400)
             & (cc[:, 2] > -300) & (cc[:, 2] < -100))
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    return MagneticsMethod(inducing_field=field).make_simulation(mesh, survey).dpred(block * 0.02)


def test_the_reduction_matches_a_vertical_field(tmp_path):
    locs = _stations()
    _write_csv(tmp_path / "m.csv", locs, _tmi(locs, FIELD))
    pole = _tmi(locs, (FIELD[0], 90.0, 0.0))
    kept = locs[::7]                                    # the job's (thinned) stations
    values, kwargs, info = worker._reduce_to_pole(["m.csv"], str(tmp_path), "magnetics", "tmi",
                                                  None, kept, {"inducing_field": FIELD})
    assert np.corrcoef(values, pole[::7])[0, 1] > 0.99
    assert values.max() == pytest.approx(pole[::7].max(), rel=0.1)
    assert kwargs["inducing_field"] == (FIELD[0], 90.0, 0.0)
    assert info["field_inclination"] == 60.0 and info["damping"] == 0.0


def test_refused_where_it_does_not_apply(tmp_path):
    locs = _stations(100.0)
    _write_csv(tmp_path / "m.csv", locs, _tmi(locs, FIELD))
    args = (["m.csv"], str(tmp_path), "magnetics", "tmi", None, locs)
    with pytest.raises(ValueError, match="vector"):
        worker._reduce_to_pole(*args, {"inducing_field": FIELD, "magnetization": "vector"})
    with pytest.raises(ValueError, match="inducing field"):
        worker._reduce_to_pole(*args, {})
    with pytest.raises(ValueError, match="total-field"):
        worker._reduce_to_pole(["m.csv"], str(tmp_path), "gravity", "gz", None, locs, {})


def test_a_job_inverts_the_reduced_residual(tmp_path):
    locs = _stations(100.0)
    _write_csv(tmp_path / "m.csv", locs, _tmi(locs, FIELD))
    ds = {"type": "magnetic", "method": "magnetics", "files": ["m.csv"], "noise_pct": 0.05,
          "noise_floor": 0.5, "component": "tmi", "method_kwargs": {"inducing_field": list(FIELD)},
          "rtp": True, "regional": {"method": "butterworth", "cutoff_m": 2500}}
    params = {"method_type": "magnetics", "inversion_mode": "single", "datasets": [ds],
              "topography": {"flat_elevation": 0}, "param_mode": "manual", "regularization_type": "l2",
              "max_iter": 4, "mesh_type": "tensor", **SMALL_MESH}
    result = worker.run_data_pipeline(params, str(tmp_path))
    info = result["datasets"][0]
    assert info["rtp"]["field_inclination"] == 60.0 and info["regional"]["method"] == "butterworth"
    # the data the inversion fitted are the reduced residual: positive over the block, centred on it
    obs, xy = result["data"]["observed"], result["data"]["locations"][:, :2]
    top = xy[np.argmax(obs)]
    assert 150 <= top[0] <= 450 and 150 <= top[1] <= 450
    json.dumps(worker._jsonable(info))      # the summary goes into result.json
