"""A DEM for any area, downloaded: SRTM 1 arc-second, else ETOPO 2022 (15 arc-seconds).

``build_dem`` makes a GeoTIFF over a box given in the data's projected CRS, in that
CRS, ready for a job's ``topography`` (the upload page treats it like a dropped DEM):

* SRTM (about 30 m, 56° S to 60° N): the 1° tiles of the AWS terrain-tile archive
  (``skadi``, public, no key), downloaded once and cached.  A tile that does not
  exist is sea; voids inside tiles are filled from their surroundings.
* ETOPO 2022 (about 450 m, global, land and sea floor): the NOAA NCEI 15° tiles,
  read remotely by window (they are tiled GeoTIFFs), when SRTM does not cover the box
  or ``source="etopo"``.

The tiles are resampled (averaged) onto a grid in the data's CRS whose pixel is the
source's (30 m or 450 m) times the smallest whole factor that keeps the longer side
within ``max_pixels``; ``margin_m`` widens the box (the mesh's padding needs ground
too).  Heights are metres above the geoid (EGM96 for SRTM, EGM2008 for ETOPO).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{name}.hgt.gz"
# 15° tiles named by their north-west corner: N30E075 covers 15–30° N, 75–90° E
ETOPO_URL = ("https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/"
             "15s_surface_elev_gtif/ETOPO_2022_v1_15s_{name}_surface.tif")
SRTM_LAT = (-56.0, 60.0)
SRTM_PIXEL_M, ETOPO_PIXEL_M = 30.0, 450.0
MAX_SRTM_TILES = 16
NODATA = -32768


def _cache_dir() -> Path:
    path = Path(os.environ.get("GEOINV3D_DEM_DIR") or Path.home() / ".geoinv3d" / "dem")
    path.mkdir(parents=True, exist_ok=True)
    return path


def lonlat_bounds(bounds, crs: str) -> tuple[float, float, float, float]:
    """(west, east, south, north) in degrees of a (west, east, south, north) box in ``crs``."""
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(crs, "EPSG:4326", bounds[0], bounds[2], bounds[1], bounds[3],
                                  densify_pts=21)
    return w, e, s, n


def srtm_tiles(lon0, lon1, lat0, lat1) -> list[str]:
    """Names (e.g. "N15E076") of the 1° SRTM tiles covering a box in degrees."""
    names = []
    for lat in range(math.floor(lat0), math.floor(np.nextafter(lat1, -np.inf)) + 1):
        for lon in range(math.floor(lon0), math.floor(np.nextafter(lon1, -np.inf)) + 1):
            names.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return names


def etopo_tiles(lon0, lon1, lat0, lat1) -> list[str]:
    """Names (north-west corners, e.g. "N30E075") of the 15° ETOPO tiles covering a box."""
    names = []
    for top in range(math.ceil(np.nextafter(lat0, np.inf) / 15) * 15, math.ceil(lat1 / 15) * 15 + 1, 15):
        for left in range(math.floor(lon0 / 15) * 15, math.floor(np.nextafter(lon1, -np.inf) / 15) * 15 + 1, 15):
            names.append(f"{'N' if top >= 0 else 'S'}{abs(top):02d}{'E' if left >= 0 else 'W'}{abs(left):03d}")
    return names


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    """Fetch ``url`` to ``dest`` (through a .part file); False if it does not exist (404)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "GeoInv3D"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code in (403, 404):     # S3 answers 403 for a key that is not there
            return False
        raise
    os.replace(tmp, dest)
    return True


def srtm_tile(name: str, log: Callable[[str], None] = print) -> Optional[Path]:
    """The .hgt file of an SRTM tile, from the cache or downloaded; None for sea."""
    cache = _cache_dir() / "srtm"
    cache.mkdir(exist_ok=True)
    hgt, missing = cache / f"{name}.hgt", cache / f"{name}.none"
    if hgt.exists():
        return hgt
    if missing.exists():
        return None
    gz = cache / f"{name}.hgt.gz"
    log(f"[DEM] Downloading SRTM tile {name}")
    if not _download(SRTM_URL.format(ns=name[0], lat=int(name[1:3]), name=name), gz):
        missing.write_text("no SRTM tile here (sea)\n")
        return None
    with gzip.open(gz, "rb") as fin, open(hgt.with_suffix(".tmp"), "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    os.replace(hgt.with_suffix(".tmp"), hgt)
    gz.unlink(missing_ok=True)
    return hgt


def _read_hgt(path: Path):
    """An SRTM tile: (heights, transform) in degrees; 3601 x 3601 big-endian int16,
    pixel centres on whole arc-seconds, north row first."""
    from rasterio.transform import from_origin
    n = int(round(math.sqrt(path.stat().st_size // 2)))
    z = np.fromfile(path, dtype=">i2").reshape(n, n).astype(np.float32)
    z[z <= -32767] = np.nan
    name = path.stem
    lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
    step = 1.0 / (n - 1)
    return z, from_origin(lon - step / 2, lat + 1 + step / 2, step, step)


def _grid(bounds, pixel: float, max_pixels: int, base: float):
    """The output grid: pixel ``base`` times a whole factor, the longer side <= max_pixels."""
    w, e, s, n = bounds
    k = max(1, math.ceil(max(e - w, n - s) / (base * max_pixels)))
    px = max(pixel, base * k)
    nx, ny = max(2, math.ceil((e - w) / px)), max(2, math.ceil((n - s) / px))
    from rasterio.transform import from_origin
    return from_origin(w, n, px, px), nx, ny, px


def build_dem(bounds, crs: str, source: str = "auto", margin_m: float = 0.0,
              max_pixels: int = 3000, pixel_m: float = 0.0,
              log: Callable[[str], None] = print) -> dict:
    """A GeoTIFF DEM over ``bounds`` = (west, east, south, north) in ``crs`` (projected).

    Returns {"path", "id", "source", "tiles", "pixel_m", "shape", "elev_min", "elev_max",
    "sea_tiles", "bounds"}.  The same request returns the cached file.  ``source`` is
    "auto" (SRTM where it covers the box, else ETOPO), "srtm" or "etopo".
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if source not in ("auto", "srtm", "etopo"):
        raise ValueError(f"Unknown DEM source '{source}' (auto, srtm or etopo)")
    w, e, s, n = (float(v) for v in bounds)
    if not (e > w and n > s):
        raise ValueError("The DEM box needs west < east and south < north")
    box = (w - margin_m, e + margin_m, s - margin_m, n + margin_m)
    lon0, lon1, lat0, lat1 = lonlat_bounds(box, crs)
    srtm = srtm_tiles(lon0, lon1, lat0, lat1)
    use = source
    if source == "auto":
        use = "srtm" if SRTM_LAT[0] <= lat0 and lat1 <= SRTM_LAT[1] and len(srtm) <= MAX_SRTM_TILES else "etopo"
    if use == "srtm" and len(srtm) > MAX_SRTM_TILES:
        raise ValueError(f"The box needs {len(srtm)} SRTM tiles (at most {MAX_SRTM_TILES}): choose a "
                         "smaller area (a window in the Area step) or use ETOPO")
    base = SRTM_PIXEL_M if use == "srtm" else ETOPO_PIXEL_M
    transform, nx, ny, px = _grid(box, pixel_m, max_pixels, base)

    key = hashlib.sha1(json.dumps([crs, [round(v, 1) for v in box], use, px]).encode()).hexdigest()[:12]
    out = _cache_dir() / f"dem_{key}.tif"
    meta_path = out.with_suffix(".json")
    if out.exists() and meta_path.exists():
        return {**json.loads(meta_path.read_text()), "path": str(out), "cached": True}

    dest = np.full((ny, nx), np.nan, dtype=np.float32)
    used, sea = [], []

    def add(z, src_transform, src_crs):
        part = np.full_like(dest, np.nan)
        reproject(z, part, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.average if px > base * 1.5 else Resampling.bilinear)
        fill = np.isnan(dest) & np.isfinite(part)
        dest[fill] = part[fill]

    if use == "srtm":
        for name in srtm:
            path = srtm_tile(name, log)
            if path is None:
                sea.append(name)
                continue
            z, t = _read_hgt(path)
            add(z, t, "EPSG:4326")
            used.append(name)
        # sea tiles and the coast beyond the last land pixel are at sea level
        covered = np.isfinite(dest)
        if not used:
            raise ValueError("No SRTM tile covers this area: it is all sea (use ETOPO for the sea floor)")
    else:
        from rasterio.windows import from_bounds
        for name in etopo_tiles(lon0, lon1, lat0, lat1):
            url = "/vsicurl/" + ETOPO_URL.format(name=name)
            log(f"[DEM] Reading ETOPO 2022 tile {name}")
            with rasterio.open(url) as src:
                pad = 2 * src.res[0]
                win = from_bounds(max(lon0 - pad, src.bounds.left), max(lat0 - pad, src.bounds.bottom),
                                  min(lon1 + pad, src.bounds.right), min(lat1 + pad, src.bounds.top),
                                  src.transform).round_offsets().round_lengths()
                if win.width < 1 or win.height < 1:
                    continue
                z = src.read(1, window=win).astype(np.float32)
                if src.nodata is not None:
                    z[z == src.nodata] = np.nan
                add(z, src.window_transform(win), "EPSG:4326")
            used.append(name)
        covered = np.isfinite(dest)
    if not covered.any():
        raise ValueError("The DEM sources returned no heights for this area")
    holes = ~covered
    if holes.any():
        if use == "srtm" and sea:
            dest[holes] = 0.0      # sea tiles (and voids next to them) are at sea level
        else:
            from rasterio.fill import fillnodata
            dest = fillnodata(dest, mask=covered.astype(np.uint8), max_search_distance=200)
            dest[np.isnan(dest)] = 0.0

    z16 = np.round(dest).astype(np.int16)
    with rasterio.open(out.with_suffix(".tmp.tif"), "w", driver="GTiff", width=nx, height=ny, count=1,
                       dtype="int16", crs=crs, transform=transform, nodata=NODATA,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(z16, 1)
        dst.update_tags(source=("SRTM 1 arc-second (AWS terrain tiles)" if use == "srtm"
                                else "ETOPO 2022 15 arc-second (NOAA NCEI)"), tiles=",".join(used))
    os.replace(out.with_suffix(".tmp.tif"), out)
    meta = {"id": key, "source": use, "tiles": used, "sea_tiles": sea, "pixel_m": px,
            "shape": [ny, nx], "elev_min": float(np.min(z16)), "elev_max": float(np.max(z16)),
            "bounds": list(box), "crs": crs,
            "file_name": f"{'srtm30' if use == 'srtm' else 'etopo2022'}_dem_{key}.tif"}
    meta_path.write_text(json.dumps(meta))
    return {**meta, "path": str(out), "cached": False}


def cached_dem(dem_id: str) -> Optional[dict]:
    """A DEM made earlier, by its id."""
    if not dem_id.isalnum():
        return None
    out = _cache_dir() / f"dem_{dem_id}.tif"
    meta = out.with_suffix(".json")
    if not (out.exists() and meta.exists()):
        return None
    return {**json.loads(meta.read_text()), "path": str(out)}
