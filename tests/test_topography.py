"""Coordinates and topography in the data pipeline: lon/lat tables projected to the
job's CRS, topography from a DEM in another CRS or from the stations' own elevations,
and the ground and stations in the viewer's 3D output."""

import json
import zipfile

import numpy as np
import pytest

from geoinv3d.cloud.worker import pack_result, run_data_pipeline
from geoinv3d.io.crs import looks_geographic, project, utm_crs, working_crs
from geoinv3d.io.readers import read_station_table
from geoinv3d.viz.result_workflow import build_workflow, load_result

LON0, LAT0 = 76.5, 15.0          # Karnataka: UTM zone 43N
SMALL = {"core_cell_m": 1000.0, "core_cell_z_m": 500.0, "depth_core_m": 3000.0,
         "pad_distance_m": 2000.0}


class TestCrs:
    def test_degrees_or_metres(self):
        assert looks_geographic([76.1, 76.9], [14.8, 15.4])
        assert not looks_geographic([641000, 711000], [1634000, 1704000])
        assert not looks_geographic([10, 200], [5, 6])          # |x| > 180
        assert not looks_geographic([0, 50], [0, 1])            # 50 degrees wide: metres
        assert not looks_geographic([np.nan], [np.nan])

    def test_utm_zone(self):
        assert utm_crs([76.2, 76.8], [15.0, 15.1]) == "EPSG:32643"
        assert utm_crs([-70.0], [-33.0]) == "EPSG:32719"

    def test_projection_round_trip(self):
        x, y = project([LON0], [LAT0], "EPSG:4326", "EPSG:32643")
        assert 600e3 < x[0] < 700e3 and 1.6e6 < y[0] < 1.7e6
        lon, lat = project(x, y, "EPSG:32643", "EPSG:4326")
        assert lon[0] == pytest.approx(LON0, abs=1e-7) and lat[0] == pytest.approx(LAT0, abs=1e-7)
        nx, ny = project([np.nan, LON0], [LAT0, LAT0], "EPSG:4326", "EPSG:32643")
        assert np.isnan(nx[0]) and np.isfinite(nx[1])

    def test_working_crs_order(self):
        lonlat = [(np.array([76.5]), np.array([15.0]))]
        assert working_crs([None, "EPSG:32644"], lonlat) == "EPSG:32644"   # a grid's CRS first
        assert working_crs([None], lonlat) == "EPSG:32643"                 # else the tables' zone
        assert working_crs([], lonlat, requested="EPSG:24343") == "EPSG:24343"
        assert working_crs([], []) is None


def test_station_table_with_a_text_column(tmp_path):
    """The NGPM station file: a map-sheet column of text used to make every row unreadable."""
    p = tmp_path / "ngpm.csv"
    p.write_text("X,Y,bouguer_an,elevation,observed_g,theoretical_g,toposheet\n"
                 "76.86,14.25,-72,548,978164,978345,57B/15\n"
                 "76.99,14.26,-68,576,978164,978345,57B/15\n"
                 "76.98,14.26,-67,573,978165,978345,\n")
    pts = read_station_table(str(p))
    np.testing.assert_allclose(pts.values, [-72, -68, -67])
    np.testing.assert_allclose(pts.locations[:, 2], [548, 576, 573])
    assert pts.column_names[2] == "bouguer_an" and "toposheet" not in pts.column_names


def _stations_lonlat(n=8, step=0.009):
    """n x n stations ~1 km apart around (LON0, LAT0) over a hill; a smooth anomaly."""
    lon, lat = np.meshgrid(LON0 + step * np.arange(n), LAT0 + step * np.arange(n))
    lon, lat = lon.ravel(), lat.ravel()
    r2 = ((lon - lon.mean()) / 0.02) ** 2 + ((lat - lat.mean()) / 0.02) ** 2
    elev = 500 + 300 * np.exp(-r2)
    gz = 5 * np.exp(-r2 / 2)                       # mGal, positive down
    return lon, lat, elev, gz


def _write_table(path, cols, rows):
    path.write_text(",".join(cols) + "\n" + "\n".join(",".join(f"{v:.6f}" for v in r) for r in rows)
                    + "\n")


def _params(files, **extra):
    ds = {"type": "gravity", "method": "gravity", "files": files, "noise_pct": 0.0,
          "noise_floor": 0.2, "component": "gz"}
    p = {"method_type": "gravity", "inversion_mode": "single", "datasets": [ds],
         "data_file": files[0], "param_mode": "manual", "regularization_type": "l2",
         "max_iter": 2, "mesh_type": "tensor", **SMALL}
    p.update(extra)
    return p


class TestPipeline:
    def test_lonlat_table_with_station_elevations(self, tmp_path):
        lon, lat, elev, gz = _stations_lonlat()
        _write_table(tmp_path / "stations.csv", ["lon", "lat", "elevation", "bouguer_an"],
                     np.column_stack([lon, lat, elev, gz]))
        result = run_data_pipeline(_params(["stations.csv"], topography={"from_data": True}),
                                   str(tmp_path))
        assert result["crs"] == "EPSG:32643"
        assert result["topography"]["source"] == "stations"
        assert result["topography"]["elevation_max"] == pytest.approx(elev.max(), abs=30)
        # stations projected to metres and kept at their elevations
        locs = result["data"]["locations"]
        ex, ey = project(lon, lat, "EPSG:4326", "EPSG:32643")
        np.testing.assert_allclose(np.sort(locs[:, 0]), np.sort(ex), atol=1e-3)
        np.testing.assert_allclose(np.sort(locs[:, 2]), np.sort(elev), atol=1e-6)
        # the mesh follows the ground: cells above it are inactive
        assert result["active_cells"].sum() < result["n_cells"]

        # the viewer: ground, stations at their heights, and the relief layers kept
        path = pack_result(result, str(tmp_path / "result.zip"))
        with zipfile.ZipFile(path) as zf:
            assert "topography.npz" in zf.namelist()
            assert "topography_grid" not in json.loads(zf.read("result.json"))
        run = load_result(path)
        run["_name"] = "topo"
        wf = build_workflow([run])
        m3 = next(n for n in wf["nodes"] if n["type"] == "RegularizedInversionNode")["output"]["model_3d"]
        assert len(m3["surface"]["z"]) == len(m3["surface"]["y"]) == 101
        assert max(max(r) for r in m3["surface"]["z"]) == pytest.approx(elev.max(), abs=30)
        np.testing.assert_allclose(sorted(m3["stations"]["z"]), np.sort(elev), atol=0.05)
        z_top = max(m3["z_edges"])
        n_relief = int(np.ceil((z_top - elev.min()) / 500.0))
        assert m3["nz"] >= 3000 / 500 + n_relief - 1   # core plus the relief layers
        assert z_top >= elev.max() - 1

    def test_utm_grid_with_lonlat_topography(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        lon, lat, elev, gz = _stations_lonlat()
        x0, y0 = project([LON0], [LAT0], "EPSG:4326", "EPSG:32643")
        # a 500 m grid of the anomaly in UTM 43N (no elevations)
        nx, ny, h = 16, 16, 500.0
        with rasterio.open(tmp_path / "ba.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                           dtype="float32", crs="EPSG:32643",
                           transform=from_origin(x0[0], y0[0] + ny * h, h, h)) as dst:
            dst.write(np.fromfunction(lambda i, j: np.sin(i / 3) + np.cos(j / 4), (ny, nx),
                                      dtype=float).astype("float32"), 1)
        _write_table(tmp_path / "elev.csv", ["X", "Y", "elevation"], np.column_stack([lon, lat, elev]))
        result = run_data_pipeline(_params(["ba.tif"], topography={"file": "elev.csv"}), str(tmp_path))
        assert result["crs"] == "EPSG:32643" and result["topography"]["source"] == "dem"
        # grid nodes draped on the ground from the projected station elevations
        z = result["data"]["locations"][:, 2]
        assert 500 - 1 < z.min() and z.max() < 800 + 1 and np.ptp(z) > 100

    def test_geographic_dem_with_utm_data(self, tmp_path):
        """An SRTM-like DEM in EPSG:4326 is sampled in its own coordinates; the job stays in UTM."""
        import rasterio
        from rasterio.transform import from_origin

        lon, lat, elev, gz = _stations_lonlat()
        ex, ey = project(lon, lat, "EPSG:4326", "EPSG:32643")
        _write_table(tmp_path / "grav.csv", ["x", "y", "gz"], np.column_stack([ex, ey, gz]))
        d = 0.002
        dem_lon = np.arange(LON0 - 0.05, LON0 + 0.12, d)
        dem_lat = np.arange(LAT0 + 0.12, LAT0 - 0.05, -d)
        with rasterio.open(tmp_path / "dem.tif", "w", driver="GTiff", width=len(dem_lon),
                           height=len(dem_lat), count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(dem_lon[0] - d / 2, dem_lat[0] + d / 2, d, d)) as dst:
            glon, glat = np.meshgrid(dem_lon, dem_lat)
            dst.write((300 + 1000 * (glon - LON0)).astype("float32"), 1)   # rises eastwards
        result = run_data_pipeline(_params(["grav.csv"], topography={"file": "dem.tif"}),
                                   str(tmp_path))
        assert result["crs"] is None or result["crs"].startswith("EPSG:326")
        locs = result["data"]["locations"]
        expect = 300 + 1000 * (lon - LON0)
        order_got, order_exp = np.argsort(locs[:, 0]), np.argsort(ex)
        np.testing.assert_allclose(locs[order_got, 2], expect[order_exp], atol=3.0)

    def test_lonlat_needs_a_crs_only_when_nothing_gives_one(self, tmp_path):
        lon, lat, elev, gz = _stations_lonlat()
        _write_table(tmp_path / "s.csv", ["lon", "lat", "gz"], np.column_stack([lon, lat, gz]))
        # stated explicitly: used as given
        result = run_data_pipeline(_params(["s.csv"], crs="EPSG:32643"), str(tmp_path))
        assert result["crs"] == "EPSG:32643" and result["data"]["locations"][:, 0].min() > 1e5
