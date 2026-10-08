"""Inputs of the regularization trials on the Block-8 5 km window: the TMI grid around the
window, and the ground from the Copernicus GLO-90 tiles.

    py examples/output/block8_regularization/scripts/prepare_window.py [DESKTOP]

Reads, from DESKTOP (default ~/OneDrive - Imperial College London/Desktop):
  karnataka_ap_magnetic/GRIDS/GEOTIFF/TAIL_TMI_GE.tiff   TMI, 37.5 m grid, flown 80 m above ground
  karnataka_ap_gravity/DEM/Copernicus_DSM_COG_30_*.tif   GLO-90 tiles (lon/lat)

Writes to data/inputs/:
  tmi_window.tif   the TMI grid over the window plus MARGIN (EPSG:32643, 37.5 m, NaN = no data)
  dem_glo90.tif    the ground over the window plus DEM_MARGIN, 45 m (bilinear from the 90 m DEM)
  window.json      the window and what the grid holds there

The window is the one of the user's Block-8 job of 2 October (LOGBOOK): 5 km on the iron-formation
ridges, -4,132 to +8,864 nT; its corners were found again from those extremes in the grid.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "inputs"
DESKTOP = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "OneDrive - Imperial College London" / "Desktop"
CRS = "EPSG:32643"
WINDOW = (659650.0, 664650.0, 1664550.0, 1669550.0)     # west, east, south, north
MARGIN = 6000.0           # TMI kept around the window (for figures and the edge checks)
DEM_MARGIN = 8000.0       # the mesh's padding (2 km) and more
DEM_RES = 45.0


def crop_tmi():
    with rasterio.open(DESKTOP / "karnataka_ap_magnetic" / "GRIDS" / "GEOTIFF" / "TAIL_TMI_GE.tiff") as src:
        t = src.transform
        west, east, south, north = WINDOW
        c0 = int(np.floor((west - MARGIN - t.c) / t.a))
        c1 = int(np.ceil((east + MARGIN - t.c) / t.a))
        r0 = int(np.floor((t.f - (north + MARGIN)) / -t.e))
        r1 = int(np.ceil((t.f - (south - MARGIN)) / -t.e))
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        v = src.read(1, window=win).astype(np.float32)
        v[~np.isfinite(v) | (np.abs(v) > 1e30)] = np.nan
        tw = src.window_transform(win)
    with rasterio.open(OUT / "tmi_window.tif", "w", driver="GTiff", height=v.shape[0], width=v.shape[1],
                       count=1, dtype="float32", crs=CRS, transform=tw, nodata=np.nan) as dst:
        dst.write(v, 1)
    x = tw.c + (np.arange(v.shape[1]) + 0.5) * tw.a
    y = tw.f + (np.arange(v.shape[0]) + 0.5) * tw.e
    mx, my = (x >= WINDOW[0]) & (x <= WINDOW[1]), (y >= WINDOW[2]) & (y <= WINDOW[3])
    w = v[np.ix_(my, mx)]
    return {"nodes": [int(mx.sum()), int(my.sum())], "n": int(np.isfinite(w).sum()),
            "min": float(np.nanmin(w)), "max": float(np.nanmax(w)),
            "p05": float(np.nanpercentile(w, 5)), "p95": float(np.nanpercentile(w, 95)),
            "east_edge": [float(np.nanmin(w[:, -1])), float(np.nanmax(w[:, -1]))]}


def dem():
    tiles = sorted(glob.glob(str(DESKTOP / "karnataka_ap_gravity" / "DEM" / "Copernicus_DSM_COG_30_*.tif")))
    if not tiles:
        raise FileNotFoundError("the GLO-90 tiles are needed in karnataka_ap_gravity/DEM")
    srcs = [rasterio.open(f) for f in tiles]
    mosaic, transform = merge(srcs)
    west, east = WINDOW[0] - DEM_MARGIN, WINDOW[1] + DEM_MARGIN
    south, north = WINDOW[2] - DEM_MARGIN, WINDOW[3] + DEM_MARGIN
    nx, ny = int(round((east - west) / DEM_RES)), int(round((north - south) / DEM_RES))
    z = np.full((ny, nx), np.nan, np.float32)
    dst_t = from_origin(west, north, DEM_RES, DEM_RES)
    reproject(mosaic[0], z, src_transform=transform, src_crs=srcs[0].crs, dst_transform=dst_t,
              dst_crs=CRS, resampling=Resampling.bilinear, dst_nodata=np.nan)
    if np.isnan(z).any():
        raise ValueError(f"{int(np.isnan(z).sum())} DEM cells are empty")
    with rasterio.open(OUT / "dem_glo90.tif", "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="float32", crs=CRS, transform=dst_t) as dst:
        dst.write(z, 1)
    x = west + (np.arange(nx) + 0.5) * DEM_RES
    y = north - (np.arange(ny) + 0.5) * DEM_RES
    inside = z[np.ix_((y >= WINDOW[2]) & (y <= WINDOW[3]), (x >= WINDOW[0]) & (x <= WINDOW[1]))]
    return {"source": "Copernicus GLO-90 (DSM, EGM2008), bilinear to 45 m", "window_min": float(inside.min()),
            "window_max": float(inside.max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"crs": CRS, "window": list(WINDOW), "tmi": crop_tmi(), "dem": dem()}
    (OUT / "window.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
