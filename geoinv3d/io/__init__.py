"""File I/O: model and data persistence, multi-format readers."""

from .formats import save_model_npy, load_model_npy
from .readers import (
    GridData,
    PointData,
    read_geotiff,
    read_surfer_grd,
    read_geojson,
    read_csv,
    read_esri_ascii,
    read_ubc_obs,
    read_station_table,
    read_npz_points,
    read_auto,
    detect_format,
)

__all__ = [
    "save_model_npy", "load_model_npy",
    "GridData", "PointData",
    "read_geotiff", "read_surfer_grd", "read_geojson", "read_csv",
    "read_esri_ascii", "read_ubc_obs", "read_station_table", "read_npz_points",
    "read_auto", "detect_format",
]
