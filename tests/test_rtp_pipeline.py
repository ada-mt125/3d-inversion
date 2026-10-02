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


def _grid_tif(path, locs, values, step):
    """The values as a GeoTIFF grid (no elevations, like an aeromagnetic grid)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    xs, ys = np.unique(locs[:, 0]), np.unique(locs[:, 1])
    z = values.reshape(len(ys), len(xs))[::-1]       # north row first
    with rasterio.open(path, "w", driver="GTiff", width=len(xs), height=len(ys), count=1,
                       dtype="float32", crs="EPSG:32643",
                       transform=from_origin(xs[0] - step / 2, ys[-1] + step / 2, step, step)) as dst:
        dst.write(z.astype("float32"), 1)


def test_continued_upwards_before_thinning(tmp_path):
    locs = _stations(50.0)
    low = locs.copy(); low[:, 2] = 80.0          # flown 80 m above flat ground
    high = locs.copy(); high[:, 2] = 600.0
    _write_csv(tmp_path / "m.csv", low, _tmi(low, FIELD))
    kept = locs[::5]
    values, kwargs, rtp_info, cont = worker._full_grid_filters(
        ["m.csv"], str(tmp_path), "magnetics", "tmi", None, kept, {"inducing_field": FIELD},
        continue_by=520.0)
    expected = _tmi(high, FIELD)[::5]
    assert np.corrcoef(values, expected)[0, 1] > 0.995 and rtp_info is None
    assert values.max() == pytest.approx(expected.max(), rel=0.05) and cont["by_m"] == 520.0
    with pytest.raises(ValueError, match="above"):
        worker._continuation_distance({"station_height": 80, "continue_to_m": 50}, {})


def test_a_job_on_a_continued_grid(tmp_path):
    locs = _stations(50.0)
    low = locs.copy(); low[:, 2] = 80.0
    _grid_tif(tmp_path / "m.tif", locs, _tmi(low, FIELD), 50.0)
    ds = {"type": "magnetic", "method": "magnetics", "files": ["m.tif"], "noise_pct": 0.05,
          "noise_floor": 0.5, "component": "tmi", "method_kwargs": {"inducing_field": list(FIELD)},
          "station_height": 80.0, "continue_to_m": 300.0, "decimate_spacing_m": 200.0}
    params = {"method_type": "magnetics", "inversion_mode": "single", "datasets": [ds],
              "topography": {"flat_elevation": 0}, "param_mode": "manual", "regularization_type": "l2",
              "max_iter": 3, "mesh_type": "tensor", "crs": "EPSG:32643", **SMALL_MESH}
    result = worker.run_data_pipeline(params, str(tmp_path))
    info = result["datasets"][0]
    assert info["continuation"]["by_m"] == 220.0 and info["continuation"]["grid_spacing_m"] == 50.0
    z = result["data"]["locations"][:, 2]
    assert np.allclose(z, 300.0)                     # at the height they were continued to
    assert result["data"]["observed"].size < locs.shape[0] / 10   # thinned after continuing


def test_the_field_from_igrf_for_the_job(tmp_path):
    """No inducing field given: the worker computes IGRF-14 at the stations' median, on the
    survey date, or on DEFAULT_IGRF_DATE when the date is not known."""
    pytest.importorskip("rasterio")
    locs = _stations(100.0) + np.array([676000.0, 1669000.0, 0.0])      # in the Karnataka study area
    _write_csv(tmp_path / "m.csv", locs, _tmi(locs - np.array([676000.0, 1669000.0, 0.0]), FIELD))
    for date, expected in ((None, worker.DEFAULT_IGRF_DATE), ("2022-06-30", "2022-06-30")):
        ds = {"type": "magnetic", "method": "magnetics", "files": ["m.csv"], "noise_pct": 0.05,
              "noise_floor": 0.5, "component": "tmi", "igrf": {"date": date}}
        params = {"method_type": "magnetics", "inversion_mode": "single", "datasets": [ds],
                  "topography": {"flat_elevation": 0}, "param_mode": "manual", "regularization_type": "l2",
                  "max_iter": 2, "mesh_type": "tensor", "crs": "EPSG:32643", **SMALL_MESH}
        info = worker.run_data_pipeline(params, str(tmp_path))["datasets"][0]["igrf"]
        assert info["date"] == expected and info["date_known"] == (date is not None)
        assert info["I"] == pytest.approx(19.3, abs=0.4) and info["D_grid"] == pytest.approx(-1.35, abs=0.3)
    with pytest.raises(ValueError, match="coordinate system"):
        worker._igrf_field({}, locs, None, {}, {})
