"""Downloaded DEMs: tile names, the output grid, and a mosaic of (synthetic) SRTM tiles."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from geoinv3d.io import dem   # noqa: E402


def test_tile_names():
    assert dem.srtm_tiles(76.2, 77.0, 15.1, 15.9) == ["N15E076"]          # an east edge on 77 stays out
    assert dem.srtm_tiles(-0.5, 0.5, -0.5, 0.5) == ["S01W001", "S01E000", "N00W001", "N00E000"]
    # ETOPO tiles are named by their north-west corner: N30E075 covers 15-30 N
    assert dem.etopo_tiles(76, 77, 15.5, 16) == ["N30E075"]
    assert dem.etopo_tiles(76, 77, 14.5, 15.5) == ["N15E075", "N30E075"]
    assert dem.etopo_tiles(-1, 1, -1, 1) == ["N00W015", "N00E000", "N15W015", "N15E000"]


def test_the_grid_keeps_within_its_pixel_budget():
    _, nx, ny, px = dem._grid((0, 150_000, 0, 60_000), 0.0, 3000, 30.0)
    assert px == 60.0 and nx == 2500 and ny == 1000
    assert dem._grid((0, 30_000, 0, 30_000), 0.0, 3000, 30.0)[3] == 30.0
    assert dem._grid((0, 30_000, 0, 30_000), 250.0, 3000, 30.0)[3] == 250.0


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    """SRTM tiles of 121 x 121 (30 arc-second) whose height is 100 m per degree east of 76° E,
    plus 500 m; N15E077 is sea."""
    monkeypatch.setenv("GEOINV3D_DEM_DIR", str(tmp_path / "dem"))
    made = []

    def fake(name, log=print):
        if name == "N15E077":
            return None
        lat = int(name[1:3]); lon = int(name[4:7])
        n = 121
        lons = lon + np.arange(n) / (n - 1)
        z = np.tile(500 + 100 * (lons - 76.0), (n, 1)).astype(">i2")
        path = tmp_path / f"{name}.hgt"
        z.tofile(path)
        made.append(name)
        return path

    monkeypatch.setattr(dem, "srtm_tile", fake)
    return made


def test_a_mosaic_in_the_data_crs(tiles):
    # a 40 km box in UTM 43N around 76.5 E, 15.5 N (it reaches into N15E077, the sea tile)
    from rasterio.warp import transform
    (x,), (y,) = transform("EPSG:4326", "EPSG:32643", [76.5], [15.5])
    m = dem.build_dem((x - 20_000, x + 60_000, y - 20_000, y + 20_000), "EPSG:32643", log=lambda s: None)
    assert m["source"] == "srtm" and m["sea_tiles"] == ["N15E077"] and "N15E076" in m["tiles"]
    with rasterio.open(m["path"]) as src:
        assert src.crs.to_string() == "EPSG:32643" and src.res == (30.0, 30.0)
        z = src.read(1).astype(float)
        (lon,), _ = transform("EPSG:32643", "EPSG:4326", [x], [y])
        assert next(src.sample([(x, y)]))[0] == pytest.approx(500 + 100 * (lon - 76), abs=2)
        assert next(src.sample([(x + 55_000, y)]))[0] == 0      # the sea tile is at sea level
        assert z.min() >= 0 and z.max() <= 600
    again = dem.build_dem((x - 20_000, x + 60_000, y - 20_000, y + 20_000), "EPSG:32643")
    assert again["cached"] and again["id"] == m["id"]
    assert dem.cached_dem(m["id"])["path"] == m["path"]
    assert dem.cached_dem("../x") is None


def test_bad_requests(tiles):
    with pytest.raises(ValueError):
        dem.build_dem((10, 0, 0, 10), "EPSG:32643")
    with pytest.raises(ValueError):
        dem.build_dem((0, 10, 0, 10), "EPSG:32643", source="lidar")
