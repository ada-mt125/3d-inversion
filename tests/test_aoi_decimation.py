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


# A diamond around the 7 x 7 test grid's centre (300, 300): the nodes with |dx| + |dy| <= 300
DIAMOND = [[300, -10], [610, 300], [300, 610], [-10, 300]]


class TestAoiPolygon:
    """A polygon as the area of interest (e.g. a window turned to the strike): the data are
    cropped to it, an OcTree is refined only under it, and the core is measured from it."""

    def test_parse_aoi(self):
        from geoinv3d.cloud.worker import parse_aoi
        assert parse_aoi(None) == (None, None)
        assert parse_aoi([0, 10, 0, 5]) == ([0.0, 10.0, 0.0, 5.0], None)
        box, poly = parse_aoi(DIAMOND)
        assert box == [-10.0, 610.0, -10.0, 610.0] and poly.shape == (4, 2)
        box2, poly2 = parse_aoi({"polygon": DIAMOND + [DIAMOND[0]]})     # a closed ring
        assert box2 == box and poly2.shape == (4, 2)
        for bad in ([[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]], [[0, 0], [1, np.nan], [2, 0]],
                    {"box": [0, 1, 0, 1]}, "abc"):
            with pytest.raises(ValueError, match="aoi"):
                parse_aoi(bad)

    def test_in_polygon_and_its_margin(self):
        from geoinv3d.cloud.worker import in_polygon
        x = np.array([300.0, 0.0, 600.0, 650.0])
        y = np.array([300.0, 0.0, 300.0, 300.0])
        np.testing.assert_array_equal(in_polygon(x, y, np.array(DIAMOND, float)), [True, False, True, False])
        np.testing.assert_array_equal(in_polygon(x, y, np.array(DIAMOND, float), margin=50.0),
                                      [True, False, True, True])

    @pytest.mark.parametrize("fmt", ["grid", "points"])
    def test_data_cropped_to_the_polygon(self, tmp_path, fmt, capture):  # noqa: F811
        locs = _station_grid(0.0)
        if fmt == "grid":
            _write_surfer_ascii(tmp_path / "g.grd", _synthetic("gravity", locs))
            files = ["g.grd"]
        else:
            _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
            files = ["g.csv"]
        result = run_data_pipeline(_single("gravity", files, aoi=DIAMOND), str(tmp_path))
        kept = capture["task"].station_locations
        assert len(kept) == 25                                   # |i| + |j| <= 3 of the 7 x 7 nodes
        assert np.all(np.abs(kept[:, 0] - 300) + np.abs(kept[:, 1] - 300) <= 300 + 1e-9)
        assert result["aoi"]["polygon"] == [[float(a), float(b)] for a, b in DIAMOND]
        assert result["aoi"]["box"] == [-10.0, 610.0, -10.0, 610.0]

    def test_the_octree_is_refined_under_the_polygon_alone(self):
        from geoinv3d.cloud.worker import _build_octree_mesh, in_polygon
        flat = lambda x, y: np.zeros(np.shape(x))                # noqa: E731
        # a 4 km square turned by 45 degrees in its 5.66 km box, 100 m cells
        c, r = 3000.0, 2828.4
        poly = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]])
        args = ((c - r, c + r, c - r, c + r), flat, 100.0, 50.0, 1000.0, 500.0, [2, 2, 2])
        whole = _build_octree_mesh(*args).to_discretize()
        turned = _build_octree_mesh(*args, polygon=poly).to_discretize()
        n_whole, n_turned = int((whole.cell_centers[:, 2] < 0).sum()), int((turned.cell_centers[:, 2] < 0).sum())
        assert n_turned < 0.65 * n_whole                         # half the area refined
        fine = turned.h_gridded[:, 0] <= 100.0 + 1e-6
        cc = turned.cell_centers[fine]
        # the finest cells lie under the polygon and its margin (two cells, then the
        # refinement's own padding of a few cells)
        assert np.all(in_polygon(cc[:, 0], cc[:, 1], poly, margin=800.0))
        assert in_polygon(cc[:, 0], cc[:, 1], poly).mean() > 0.6

    def test_the_core_is_the_polygon(self):
        from discretize import TensorMesh
        from geoinv3d.cloud.worker import _outside_core_parts, _outside_core_share
        mesh = TensorMesh([np.full(8, 100.0), np.full(8, 100.0), np.full(4, 100.0)], origin=(0, 0, -400))
        model = np.zeros(mesh.n_cells)
        corner = np.argmin(np.hypot(mesh.cell_centers[:, 0] - 50, mesh.cell_centers[:, 1] - 50)
                           + np.abs(mesh.cell_centers[:, 2] + 50))
        model[corner] = 1.0                                      # in the box's corner, off the diamond
        poly = np.array([[400, 0], [800, 400], [400, 800], [0, 400]], float)
        extent, z_bottom = (0, 800, 0, 800), -400.0
        assert _outside_core_share(mesh, None, model, extent, 0.0, z_bottom) == 0.0
        assert _outside_core_share(mesh, None, model, extent, 0.0, z_bottom, poly) == 1.0
        assert _outside_core_parts(mesh, None, model, extent, 0.0, z_bottom, poly) == (1.0, 0.0)
