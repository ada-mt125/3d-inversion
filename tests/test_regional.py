"""Regional-field removal (polynomial trend surface) and its use in the pipeline."""

import numpy as np
import pytest

from geoinv3d.cloud.worker import run_data_pipeline
from geoinv3d.methods.regional import fit_trend, remove_regional
from tests.test_data_pipeline import SMALL_MESH, _single, _station_grid, _synthetic, _write_csv


@pytest.fixture
def xy():
    xx, yy = np.meshgrid(np.linspace(600e3, 700e3, 21), np.linspace(1.6e6, 1.7e6, 21))
    return np.column_stack([xx.ravel(), yy.ravel()])   # UTM-sized coordinates


class TestTrendSurface:
    def test_plane_is_removed_exactly(self, xy):
        d = -83.0 + 2e-4 * xy[:, 0] - 1e-4 * xy[:, 1]
        residual, info = remove_regional(xy, d, {"method": "polynomial", "order": 1})
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)
        assert info["order"] == 1 and info["data_std_after"] < 1e-8

    def test_quadratic_needs_order_two(self, xy):
        u = (xy[:, 0] - 650e3) / 50e3
        d = 5 + 3 * u + 4 * u ** 2
        assert np.abs(d - fit_trend(xy, d, 2)).max() < 1e-8
        assert np.abs(d - fit_trend(xy, d, 1)).max() > 0.5   # a plane misses the curvature

    def test_local_anomaly_survives(self, xy):
        r2 = (xy[:, 0] - 650e3) ** 2 + (xy[:, 1] - 1.65e6) ** 2
        anomaly = 10 * np.exp(-r2 / (2 * 8e3 ** 2))
        d = anomaly - 83 + 3e-4 * (xy[:, 0] - 650e3)
        residual, _ = remove_regional(xy, d, {"method": "polynomial", "order": 1})
        assert np.corrcoef(residual, anomaly)[0, 1] > 0.98
        assert residual.max() - residual.min() == pytest.approx(10, rel=0.1)

    def test_mean_and_none(self, xy):
        d = np.arange(len(xy), dtype=float)
        residual, info = remove_regional(xy, d, "mean")
        assert residual.mean() == pytest.approx(0) and info["order"] == 0
        same, info = remove_regional(xy, d, None)
        assert same is d and info is None
        assert remove_regional(xy, d, {"method": "none"})[1] is None

    def test_bad_spec(self, xy):
        with pytest.raises(ValueError, match="regional method"):
            remove_regional(xy, np.zeros(len(xy)), {"method": "fft"})


class TestPipelineRegional:
    def test_trend_is_removed_and_recorded(self, tmp_path):
        locs = _station_grid(0.0)
        plane = -80.0 + 0.01 * locs[:, 0]
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs) + plane)
        params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type="l2",
                         max_iter=3,
                         dataset={"regional": {"method": "polynomial", "order": 1}})
        result = run_data_pipeline(params, str(tmp_path))
        info = result["datasets"][0]["regional"]
        assert info["method"] == "polynomial" and info["order"] == 1
        assert info["data_std_after"] < info["data_std_before"]
        data = result["data"]
        np.testing.assert_allclose(data["observed"] + data["regional"],
                                   _synthetic("gravity", locs) + plane, rtol=1e-6)
        assert abs(data["observed"].mean()) < 1e-6   # a plane fit also removes the mean
