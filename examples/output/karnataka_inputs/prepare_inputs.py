"""Inputs of the Karnataka inversions with terrain: DEM, complete Bouguer anomaly, TMI at 1 km.

    py examples/output/karnataka_inputs/prepare_inputs.py [DESKTOP]

Reads, from DESKTOP (default ~/OneDrive - Imperial College London/Desktop):
  karnataka_ap_gravity/DEM/Copernicus_DSM_COG_30_*.tif   Copernicus GLO-90 tiles (lon/lat)
  karnataka_ap_gravity/GEOTIFF/NGPM_BA.tiff              simple Bouguer anomaly, 500 m grid
  karnataka_ap_gravity/ASCII/combined_NGPM_gravity.csv   the stations behind that grid
  karnataka_ap_magnetic/GRIDS/GEOTIFF/TAIL_TMI_GE.tiff   TMI anomaly, 37.5 m grid, flown 80 m
                                                          above the ground

Writes next to this script:
  dem_utm43n_450m.tif        the ground for the mesh (EPSG:32643, 450 m means of the 90 m DEM)
  gravity_simple_1km.csv     x, y, z, gz: the NGPM grid at the 1 km nodes of the earlier runs
  gravity_complete_1km.csv   the same + terrain correction (2.67 g/cc, 50 km, from the DEM)
  magnetic_1km.csv           x, y, z, tmi: continued upwards to 1 km above the ground
  magnetic_1km_h500.csv      the same at 500 m above the ground (a test of the height)
  magnetic_1km_raw80.csv     the grid sampled as flown, 80 m above the ground (a test of aliasing)
  magnetic_1km_xy.csv        x, y, tmi at 1 km above the ground, without z (for a flat earth)
  stations_tc.csv            the terrain correction at the NGPM stations
  prep.npz, prep.json        grids for the reports' figures, and the numbers quoted in them

The Bouguer grid was interpolated from the stations, each reduced at its own height without
a terrain correction.  The complete anomaly is the smooth quantity, so the stations'
corrections are gridded the same way (thin-plate spline) and added to the grid.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from geoinv3d.io.crs import project                                   # noqa: E402
from geoinv3d.methods.continuation import upward_continue            # noqa: E402
from geoinv3d.methods.terrain import terrain_correction             # noqa: E402

DESKTOP = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "OneDrive - Imperial College London" / "Desktop"
CRS = "EPSG:32643"
AOI = (641000.0, 711000.0, 1634000.0, 1704000.0)        # west, east, south, north
DEM_RES = 90.0
DEM_BOX = (575000.0, 777020.0, 1568000.0, 1768070.0)     # AOI + 64 to 66 km: the correction's reach
TC_RADIUS = 50000.0
TC_DENSITY = 2.67
STATION_MARGIN = 10000.0                                 # stations around the AOI, for gridding
FLIGHT_HEIGHT = 80.0
MAG_HEIGHT = 1000.0                                      # receivers above the ground after continuation
MAG_MARGIN = 20000.0


def dem_utm():
    """The GLO-90 tiles as one 90 m grid in UTM 43N: x, y ascending (cell centres), z."""
    tiles = sorted(glob.glob(str(DESKTOP / "karnataka_ap_gravity" / "DEM" / "Copernicus_DSM_COG_30_*.tif")))
    if len(tiles) < 6:
        raise FileNotFoundError("the six GLO-90 tiles N14-N15 / E075-E077 are needed in karnataka_ap_gravity/DEM")
    srcs = [rasterio.open(f) for f in tiles]
    mosaic, transform = merge(srcs)
    west, east, south, north = DEM_BOX
    nx, ny = int(round((east - west) / DEM_RES)), int(round((north - south) / DEM_RES))
    z = np.full((ny, nx), np.nan, np.float32)
    reproject(mosaic[0], z, src_transform=transform, src_crs=srcs[0].crs,
              dst_transform=from_origin(west, north, DEM_RES, DEM_RES), dst_crs=CRS,
              resampling=Resampling.bilinear, dst_nodata=np.nan)
    if np.isnan(z).any():
        raise ValueError(f"{int(np.isnan(z).sum())} DEM cells are empty: the tiles do not cover the box")
    x = west + (np.arange(nx) + 0.5) * DEM_RES
    y = south + (np.arange(ny) + 0.5) * DEM_RES
    return x, y, z[::-1].astype(float)                   # rows south to north


def block_mean(z, f):
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    return z[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def nodes_1km():
    """The 1 km nodes of the earlier gravity runs (every second node of the NGPM grid in the AOI)
    and the simple Bouguer anomaly there."""
    with rasterio.open(DESKTOP / "karnataka_ap_gravity" / "GEOTIFF" / "NGPM_BA.tiff") as src:
        ba = src.read(1).astype(float)
        if src.nodata is not None:
            ba[ba == src.nodata] = np.nan
        t = src.transform
        x = t.c + (np.arange(src.width) + 0.5) * t.a
        y = t.f + (np.arange(src.height) + 0.5) * t.e
    mx, my = (x >= AOI[0]) & (x <= AOI[1]), (y >= AOI[2]) & (y <= AOI[3])
    x, y, ba = x[mx][::2], y[my][::2], ba[np.ix_(my, mx)][::2, ::2]
    xx, yy = np.meshgrid(x, y)
    return xx.ravel(), yy.ravel(), ba.ravel(), (x, y, ba)


def main():
    out = {"crs": CRS, "aoi": AOI}
    print("DEM ...")
    dx, dy, dz = dem_utm()
    surface = RegularGridInterpolator((dy, dx), dz, bounds_error=False, fill_value=None)
    in_aoi = dz[np.ix_((dy >= AOI[2]) & (dy <= AOI[3]), (dx >= AOI[0]) & (dx <= AOI[1]))]
    out["dem"] = {"source": "Copernicus GLO-90 (DSM, EGM2008)", "resolution_m": DEM_RES,
                  "aoi_min": float(in_aoi.min()), "aoi_max": float(in_aoi.max()),
                  "aoi_mean": float(in_aoi.mean()), "aoi_p01": float(np.percentile(in_aoi, 1)),
                  "aoi_p99": float(np.percentile(in_aoi, 99))}

    # the ground for the mesh: 450 m means, AOI + 30 km (beyond it the pipeline holds the edge)
    f = 5
    zc = block_mean(dz, f)
    xc = dx[:zc.shape[1] * f].reshape(-1, f).mean(axis=1)
    yc = dy[:zc.shape[0] * f].reshape(-1, f).mean(axis=1)
    kx = (xc > AOI[0] - 30000) & (xc < AOI[1] + 30000)
    ky = (yc > AOI[2] - 30000) & (yc < AOI[3] + 30000)
    zs, xs, ys = zc[np.ix_(ky, kx)], xc[kx], yc[ky]
    step = DEM_RES * f
    with rasterio.open(HERE / "dem_utm43n_450m.tif", "w", driver="GTiff", height=len(ys), width=len(xs),
                       count=1, dtype="float32", crs=CRS, compress="deflate",
                       transform=from_origin(xs[0] - step / 2, ys[-1] + step / 2, step, step)) as dst:
        dst.write(zs[::-1].astype(np.float32), 1)
    smooth = RegularGridInterpolator((yc, xc), zc, bounds_error=False, fill_value=None)

    # ── gravity: terrain correction at the stations, gridded onto the nodes ──
    print("terrain correction ...")
    st = pd.read_csv(DESKTOP / "karnataka_ap_gravity" / "ASCII" / "combined_NGPM_gravity.csv")
    sx, sy = project(st.X.values, st.Y.values, "EPSG:4326", CRS)
    m = STATION_MARGIN
    near = (sx > AOI[0] - m) & (sx < AOI[1] + m) & (sy > AOI[2] - m) & (sy < AOI[3] + m)
    sx, sy, st = sx[near], sy[near], st[near].reset_index(drop=True)
    # repeated stations (530 of them share a position): one each, with the mean values
    st = st.assign(x=np.round(sx, 1), y=np.round(sy, 1)).groupby(["x", "y"], as_index=False)[
        ["bouguer_an", "elevation"]].mean()
    sx, sy = st.x.values, st.y.values
    tc, parts = terrain_correction(sx, sy, dx, dy, dz, density=TC_DENSITY, radius=TC_RADIUS,
                                   return_parts=True)
    h_dem = surface(np.column_stack([sy, sx]))
    inside = (sx >= AOI[0]) & (sx <= AOI[1]) & (sy >= AOI[2]) & (sy <= AOI[3])
    pd.DataFrame({"x": sx, "y": sy, "elevation": st.elevation, "dem": h_dem, "bouguer_an": st.bouguer_an,
                  "tc": tc, "tc_inner": parts[:, 0], "tc_near": parts[:, 1], "tc_middle": parts[:, 2],
                  "tc_far": parts[:, 3], "in_aoi": inside.astype(int)}
                 ).to_csv(HERE / "stations_tc.csv", index=False, float_format="%.3f")
    d = h_dem - st.elevation.values
    out["stations"] = {
        "n_used": int(len(sx)), "n_in_aoi": int(inside.sum()),
        "elevation_min": float(st.elevation[inside].min()), "elevation_max": float(st.elevation[inside].max()),
        "dem_minus_station_mean": float(d.mean()), "dem_minus_station_std": float(d.std()),
        "spacing_km": float(np.sqrt((AOI[1] - AOI[0]) * (AOI[3] - AOI[2]) / inside.sum()) / 1e3)}
    t_in = tc[inside]
    out["terrain_correction"] = {
        "density": TC_DENSITY, "radius_km": TC_RADIUS / 1e3,
        "stations": {"min": float(t_in.min()), "median": float(np.median(t_in)), "mean": float(t_in.mean()),
                     "p90": float(np.percentile(t_in, 90)), "p99": float(np.percentile(t_in, 99)),
                     "max": float(t_in.max()), "n_above_1": int((t_in > 1).sum()),
                     "n_above_0.5": int((t_in > 0.5).sum())},
        "zones_mean": {k: float(parts[inside, i].mean()) for i, k in enumerate(("inner", "near", "middle", "far"))},
        "zones_max": {k: float(parts[inside, i].max()) for i, k in enumerate(("inner", "near", "middle", "far"))}}

    gx, gy, ba, (nx1, ny1, ba2d) = nodes_1km()
    ok = np.isfinite(ba)
    spline = RBFInterpolator(np.column_stack([sx, sy]), tc, neighbors=40, kernel="thin_plate_spline")
    tc_nodes = np.clip(spline(np.column_stack([gx, gy])), 0.0, None)
    z_nodes = surface(np.column_stack([gy, gx]))
    for name, values in (("simple", ba), ("complete", ba + tc_nodes)):
        pd.DataFrame({"x": gx[ok], "y": gy[ok], "z": z_nodes[ok], "gz": values[ok]}
                     ).to_csv(HERE / f"gravity_{name}_1km.csv", index=False, float_format="%.3f")
    # the grid at the stations against the stations' own (rounded) values
    grid = RegularGridInterpolator((ny1[::-1], nx1), ba2d[::-1], bounds_error=False)
    diff = grid(np.column_stack([sy[inside], sx[inside]])) - st.bouguer_an.values[inside]
    out["gravity"] = {
        "n": int(ok.sum()), "simple_min": float(ba[ok].min()), "simple_max": float(ba[ok].max()),
        "tc_nodes": {"min": float(tc_nodes[ok].min()), "median": float(np.median(tc_nodes[ok])),
                     "mean": float(tc_nodes[ok].mean()), "p99": float(np.percentile(tc_nodes[ok], 99)),
                     "max": float(tc_nodes[ok].max()), "n_above_1": int((tc_nodes[ok] > 1).sum()),
                     "n_above_0.5": int((tc_nodes[ok] > 0.5).sum())},
        "node_elevation_min": float(z_nodes[ok].min()), "node_elevation_max": float(z_nodes[ok].max()),
        "grid_minus_station_std": float(np.nanstd(diff)), "grid_minus_station_mean": float(np.nanmean(diff))}

    # ── magnetics: upward continuation, then the same nodes ──
    print("magnetics ...")
    with rasterio.open(DESKTOP / "karnataka_ap_magnetic" / "GRIDS" / "GEOTIFF" / "TAIL_TMI_GE.tiff") as src:
        m = MAG_MARGIN
        win = from_bounds(AOI[0] - m, AOI[2] - m, AOI[1] + m, AOI[3] + m, src.transform)
        win = win.round_offsets().round_lengths()
        tmi = src.read(1, window=win).astype(float)
        if src.nodata is not None:
            tmi[tmi == src.nodata] = np.nan
        t = src.window_transform(win)
    if np.isnan(tmi).any():
        raise ValueError("the TMI window has gaps")
    res = float(t.a)
    mx = t.c + (np.arange(tmi.shape[1]) + 0.5) * res
    my = (t.f - (np.arange(tmi.shape[0]) + 0.5) * res)[::-1]
    tmi = tmi[::-1]                                      # rows south to north
    up = upward_continue(tmi, res, res, MAG_HEIGHT - FLIGHT_HEIGHT)
    up_half = upward_continue(tmi, res, res, 500.0 - FLIGHT_HEIGHT)
    field = RegularGridInterpolator((my, mx), up)
    raw = RegularGridInterpolator((my, mx), tmi)
    q = np.column_stack([gy, gx])
    v, v_raw = field(q), raw(q)
    z_mag = smooth(q) + MAG_HEIGHT
    pd.DataFrame({"x": gx, "y": gy, "z": z_mag, "tmi": v}
                 ).to_csv(HERE / "magnetic_1km.csv", index=False, float_format="%.3f")
    # for the tests: no elevations (a flat earth, receivers at a station_height), the grid
    # sampled as flown (80 m above the ground, not continued), and 500 m above the ground
    pd.DataFrame({"x": gx, "y": gy, "tmi": v}).to_csv(HERE / "magnetic_1km_xy.csv", index=False, float_format="%.3f")
    pd.DataFrame({"x": gx, "y": gy, "z": z_mag - MAG_HEIGHT + FLIGHT_HEIGHT, "tmi": v_raw}
                 ).to_csv(HERE / "magnetic_1km_raw80.csv", index=False, float_format="%.3f")
    pd.DataFrame({"x": gx, "y": gy, "z": z_mag - MAG_HEIGHT + 500.0, "tmi": RegularGridInterpolator((my, mx), up_half)(q)}
                 ).to_csv(HERE / "magnetic_1km_h500.csv", index=False, float_format="%.3f")
    core = np.ix_((my >= AOI[2]) & (my <= AOI[3]), (mx >= AOI[0]) & (mx <= AOI[1]))

    def above_nyquist(g):
        """Share of the variance at wavelengths shorter than 2 km (what 1 km sampling aliases)."""
        a = g[core] - g[core].mean()
        p = np.abs(np.fft.rfft2(a)) ** 2
        k = np.hypot(np.fft.fftfreq(a.shape[0], res)[:, None], np.fft.rfftfreq(a.shape[1], res)[None, :])
        return float(p[k > 1 / 2000.0].sum() / p.sum())

    out["magnetic"] = {
        "grid_cell_m": res, "flight_height_m": FLIGHT_HEIGHT, "receiver_height_m": MAG_HEIGHT,
        "continued_by_m": MAG_HEIGHT - FLIGHT_HEIGHT, "n": int(len(v)),
        "raw_grid": {"min": float(tmi[core].min()), "max": float(tmi[core].max()), "std": float(tmi[core].std())},
        "raw_at_nodes": {"min": float(v_raw.min()), "max": float(v_raw.max()), "std": float(v_raw.std())},
        "continued_at_nodes": {"min": float(v.min()), "max": float(v.max()), "std": float(v.std()),
                               "p01": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99))},
        "short_wavelength_share": {"80 m": above_nyquist(tmi), "500 m": above_nyquist(up_half),
                                   "1000 m": above_nyquist(up)},
        "receiver_z_min": float(z_mag.min()), "receiver_z_max": float(z_mag.max())}

    # grids for the figures: DEM and TMI over the AOI at 150 m, the node fields
    sub = 4
    np.savez_compressed(
        HERE / "prep.npz",
        dem_x=dx[(dx >= AOI[0]) & (dx <= AOI[1])], dem_y=dy[(dy >= AOI[2]) & (dy <= AOI[3])], dem=in_aoi.astype(np.float32),
        node_x=nx1, node_y=ny1, ba=ba2d.astype(np.float32),
        tc_nodes=tc_nodes.reshape(len(ny1), len(nx1)).astype(np.float32),
        z_nodes=z_nodes.reshape(len(ny1), len(nx1)).astype(np.float32),
        station_x=sx, station_y=sy, station_tc=tc, station_elevation=st.elevation.values,
        tmi_x=mx[core[1].ravel()][::sub], tmi_y=my[core[0].ravel()][::sub],
        tmi_raw=tmi[core][::sub, ::sub].astype(np.float32), tmi_500=up_half[core][::sub, ::sub].astype(np.float32),
        tmi_1000=up[core][::sub, ::sub].astype(np.float32),
        tmi_nodes=v.reshape(len(ny1), len(nx1)).astype(np.float32),
        tmi_raw_nodes=v_raw.reshape(len(ny1), len(nx1)).astype(np.float32))
    (HERE / "prep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
