"""Area of interest and data thinning (decimate_spacing_m) in the data pipeline."""

import numpy as np
import pytest

from geoinv3d.cloud.worker import grid_strides, run_data_pipeline, thin_points
from tests.test_data_pipeline import (
    _single, _station_grid, _synthetic, _write_csv, _write_surfer_ascii, capture,  # noqa: F401
)


class TestStrides:
    def test_target_spacing_gives_whole_strides_per_axis(self):
        assert grid_strides(500.0, 500.0, 1000.0) == (2, 2)
        assert grid_strides(100.0, 200.0, 400.0) == (4, 2)
        assert grid_strides(500.0, 500.0, 700.0) == (1, 1)      # rounds to the nearest stride
        assert grid_strides(500.0, 500.0, 200.0) == (1, 1)      # never finer than the grid

    def test_without_target_the_old_stride_applies(self):
        assert grid_strides(500.0, 500.0, None, stride=3) == (3, 3)
        assert grid_strides(500.0, 500.0, None) == (1, 1)


class TestThinPoints:
    def test_one_point_per_cell_first_in_file_order(self):
        xx, yy = np.meshgrid(np.arange(0, 100, 10.0), np.arange(0, 100, 10.0))
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        keep = thin_points(xy, 25.0)
        cells = {(int(x // 25), int(y // 25)) for x, y in xy[keep]}
        assert len(keep) == len(cells) == 16                     # 4 x 4 cells of 25 m
        assert np.all(np.diff(keep) > 0)                         # file order kept
        np.testing.assert_array_equal(xy[keep[0]], [0.0, 0.0])

    def test_origin_shifts_the_cells_and_no_spacing_keeps_all(self):
        xy = np.array([[4.0, 0.0], [6.0, 0.0]])
        assert len(thin_points(xy, 10.0)) == 1                   # both in [4, 14)
        assert len(thin_points(xy, 10.0, origin=(5.0, 0.0))) == 2
        assert len(thin_points(xy, 0)) == 2


class TestPipeline:
    def test_grid_aoi_and_target_spacing(self, tmp_path, capture):  # noqa: F811
        # 7 x 7 nodes every 100 m; AOI keeps x, y in [100, 500] (5 x 5), 200 m keeps 3 x 3
        _write_surfer_ascii(tmp_path / "grav.grd", _synthetic("gravity", _station_grid(0.0)))
        params = _single("gravity", ["grav.grd"], aoi=[100, 500, 100, 500],
                         dataset={"decimate_spacing_m": 200.0})
        result = run_data_pipeline(params, str(tmp_path))
        locs = capture["task"].station_locations
        assert len(locs) == 9
        assert set(np.unique(locs[:, 0])) == {100.0, 300.0, 500.0}
        dec = result["datasets"][0]["decimation"]
        assert dec["target_spacing_m"] == 200.0
        assert dec["files"]["grav.grd"]["stride"] == [2, 2]

    def test_points_thinned_per_dataset(self, tmp_path, capture):  # noqa: F811
        locs = _station_grid(0.0)                                 # 49 stations every 100 m
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        params = _single("gravity", ["g.csv"], dataset={"decimate_spacing_m": 250.0})
        result = run_data_pipeline(params, str(tmp_path))
        kept = capture["task"].station_locations
        assert len(kept) == 9                                    # 3 x 3 cells of 250 m
        info = result["datasets"][0]["decimation"]["files"]["g.csv"]
        assert info == {"kind": "points", "cell_m": 250.0, "n_before": 49}

    def test_job_wide_spacing_and_no_thinning_by_default(self, tmp_path, capture):  # noqa: F811
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        result = run_data_pipeline(_single("gravity", ["g.csv"]), str(tmp_path))
        assert len(capture["task"].station_locations) == 49
        assert result["datasets"][0]["decimation"] is None
        run_data_pipeline(_single("gravity", ["g.csv"], decimate_spacing_m=250.0), str(tmp_path))
        assert len(capture["task"].station_locations) == 9

    @pytest.mark.parametrize("aoi", [[500, 100, 0, 600], [0, 600, 0], [0, 600, 300, 300]])
    def test_bad_aoi_is_refused(self, tmp_path, aoi, capture):  # noqa: F811
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        with pytest.raises(ValueError, match="aoi"):
            run_data_pipeline(_single("gravity", ["g.csv"], aoi=aoi), str(tmp_path))
