"""Automatic data errors: a floor of 1.5 % of the data's 5-95 % spread, and 2 %."""

import numpy as np
import pytest

from geoinv3d.cloud.worker import AUTO_FLOOR_SHARE, AUTO_NOISE_PCT, auto_noise, run_data_pipeline
from tests.test_data_pipeline import _single, _station_grid, _synthetic, _write_csv


def test_the_floor_follows_the_spread_of_the_data():
    v = np.linspace(-1000.0, 1000.0, 2001)               # 5-95 %: -900 to 900
    a = auto_noise(v, "auto", "auto")
    assert a["spread"] == pytest.approx(1800.0)
    assert a["noise_floor"] == pytest.approx(AUTO_FLOOR_SHARE * 1800.0)
    assert a["noise_pct"] == AUTO_NOISE_PCT
    # an outlier does not move it (percentiles, not the range)
    assert auto_noise(np.append(v, 1e6), "auto", "auto")["noise_floor"] == pytest.approx(27.0, rel=0.01)


def test_numbers_given_are_kept():
    assert auto_noise([1.0, 2.0], 0.05, 0.5) is None                  # nothing automatic
    a = auto_noise(np.linspace(0, 100, 101), 0.03, "auto")
    assert a["noise_pct"] == 0.03 and a["noise_floor"] == pytest.approx(1.35)
    b = auto_noise(np.linspace(0, 100, 101), "auto", 4.0)
    assert b["noise_pct"] == AUTO_NOISE_PCT and b["noise_floor"] == 4.0
    assert auto_noise(np.full(10, 7.0), 0.05, "auto")["noise_floor"] > 0   # constant data


def test_a_job_with_automatic_errors_records_them(tmp_path):
    locs = _station_grid(0.0)
    data = _synthetic("gravity", locs)
    _write_csv(tmp_path / "g.csv", locs, data)
    params = _single("gravity", ["g.csv"], param_mode="manual", regularization_type="l2",
                     max_iter=3, dataset={"noise_pct": "auto", "noise_floor": "auto"})
    result = run_data_pipeline(params, str(tmp_path))
    ds = result["datasets"][0]
    p5, p95 = np.percentile(data, [5, 95])
    assert ds["noise_auto"]["spread"] == pytest.approx(p95 - p5, rel=1e-6)
    assert ds["noise_floor"] == pytest.approx(AUTO_FLOOR_SHARE * (p95 - p5), rel=1e-6)
    assert ds["noise_pct"] == AUTO_NOISE_PCT
