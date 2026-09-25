"""Multi-format geophysical data readers.

Supported formats:
    - GeoTIFF (.tif/.tiff) — raster grids (topography, gravity, magnetics)
    - Surfer GRD (.grd) — binary grid format (Surfer 6 and 7)
    - ESRI ASCII grid (.asc) — raster DEMs
    - UBC-GIF observations (.obs) — gravity / magnetic station data
    - GeoJSON (.geojson/.json) — point/polygon features (station locations)
    - CSV (.csv/.txt/.dat/.xyz) — tabular station data
    - NPY (.npy) / NPZ (.npz) — raw NumPy arrays
"""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class GridData:
    """A 2D regular grid with coordinates and metadata."""
    values: NDArray
    x: NDArray
    y: NDArray
    nx: int
    ny: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    dx: float
    dy: float
    nodata: float | None = None
    crs: str | None = None
    source_path: str = ""
    data_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PointData:
    """Point observations with coordinates."""
    locations: NDArray
    values: NDArray | None = None
    column_names: list[str] = field(default_factory=list)
    attributes: dict[str, NDArray] = field(default_factory=dict)
    crs: str | None = None
    source_path: str = ""
    data_type: str = ""
    metadata: dict = field(default_factory=dict)


def read_geotiff(path: str, band: int = 1) -> GridData:
    """Read a GeoTIFF file into a GridData object.

    Args:
        path: Path to .tif/.tiff file.
        band: Band number to read (1-indexed).

    Returns:
        GridData with values, coordinates, and CRS.
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float64)
        transform = src.transform
        ny, nx = data.shape
        nodata = src.nodata

        dx = transform.a
        dy = -transform.e
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + nx * dx
        ymin = ymax - ny * dy

        x = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
        y = np.linspace(ymax - dy / 2, ymin + dy / 2, ny)

        crs_str = str(src.crs) if src.crs else None

        if nodata is not None:
            data[data == nodata] = np.nan

    return GridData(
        values=data,
        x=x, y=y,
        nx=nx, ny=ny,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        dx=dx, dy=dy,
        nodata=nodata,
        crs=crs_str,
        source_path=str(Path(path).resolve()),
        metadata={"band": band, "format": "geotiff"},
    )


def read_surfer_grd(path: str) -> GridData:
    """Read a Surfer GRD file (version 6 binary or 7 binary).

    Supports both Surfer 6 (.grd with 'DSBB' header) and
    Surfer 7 (.grd with 'DSRB' header).
    """
    p = Path(path)
    with open(p, "rb") as f:
        tag = f.read(4)

        if tag == b"DSBB":
            return _read_surfer6(f, str(p.resolve()))
        elif tag == b"DSRB":
            return _read_surfer7(f, str(p.resolve()))
        else:
            f.seek(0)
            first_line = f.read(4).decode("ascii", errors="ignore")
            if first_line.startswith("DSAA"):
                f.seek(0)
                return _read_surfer_ascii(f, str(p.resolve()))
            raise ValueError(
                f"Unknown GRD format (header: {tag!r}). "
                f"Expected Surfer 6 (DSBB), Surfer 7 (DSRB), or ASCII (DSAA)."
            )


def _read_surfer6(f, source_path: str) -> GridData:
    """Surfer 6 binary: DSBB header."""
    nx, ny = struct.unpack("<HH", f.read(4))
    xmin, xmax = struct.unpack("<dd", f.read(16))
    ymin, ymax = struct.unpack("<dd", f.read(16))
    zmin, zmax = struct.unpack("<dd", f.read(16))

    data = np.frombuffer(f.read(nx * ny * 4), dtype="<f4")
    data = data.reshape(ny, nx).astype(np.float64)

    nodata = 1.70141e+38
    data[data >= nodata * 0.99] = np.nan

    dx = (xmax - xmin) / max(nx - 1, 1)
    dy = (ymax - ymin) / max(ny - 1, 1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    return GridData(
        values=data, x=x, y=y,
        nx=nx, ny=ny,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        dx=dx, dy=dy,
        nodata=nodata,
        source_path=source_path,
        metadata={"format": "surfer6_binary", "zmin": zmin, "zmax": zmax},
    )


def _read_surfer7(f, source_path: str) -> GridData:
    """Surfer 7 binary: DSRB header with sections."""
    _version, _size = struct.unpack("<II", f.read(8))

    tag2 = f.read(4)
    if tag2 != b"GRID":
        raise ValueError(f"Expected GRID section, got {tag2!r}")
    _grid_size = struct.unpack("<I", f.read(4))[0]

    ny, nx = struct.unpack("<II", f.read(8))
    xmin, ymin = struct.unpack("<dd", f.read(16))
    dx, dy = struct.unpack("<dd", f.read(16))
    zmin, zmax = struct.unpack("<dd", f.read(16))
    _rotation = struct.unpack("<d", f.read(8))[0]
    _blank_value = struct.unpack("<d", f.read(8))[0]

    tag3 = f.read(4)
    if tag3 != b"DATA":
        raise ValueError(f"Expected DATA section, got {tag3!r}")
    _data_size = struct.unpack("<I", f.read(4))[0]

    data = np.frombuffer(f.read(nx * ny * 8), dtype="<f8")
    data = data.reshape(ny, nx).copy()
    data[data == _blank_value] = np.nan

    xmax = xmin + (nx - 1) * dx
    ymax = ymin + (ny - 1) * dy
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    return GridData(
        values=data, x=x, y=y,
        nx=nx, ny=ny,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        dx=dx, dy=dy,
        nodata=_blank_value,
        source_path=source_path,
        metadata={"format": "surfer7_binary", "zmin": zmin, "zmax": zmax},
    )


def _read_surfer_ascii(f, source_path: str) -> GridData:
    """Surfer ASCII: DSAA header."""
    lines = f.read().decode("utf-8").splitlines()
    idx = 1
    nx, ny = map(int, lines[idx].split())
    idx += 1
    xmin, xmax = map(float, lines[idx].split())
    idx += 1
    ymin, ymax = map(float, lines[idx].split())
    idx += 1
    zmin, zmax = map(float, lines[idx].split())
    idx += 1

    values = []
    for line in lines[idx:]:
        values.extend(float(v) for v in line.split())

    data = np.array(values).reshape(ny, nx)
    nodata = 1.70141e+38
    data[data >= nodata * 0.99] = np.nan

    dx = (xmax - xmin) / max(nx - 1, 1)
    dy = (ymax - ymin) / max(ny - 1, 1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    return GridData(
        values=data, x=x, y=y,
        nx=nx, ny=ny,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        dx=dx, dy=dy,
        nodata=nodata,
        source_path=source_path,
        metadata={"format": "surfer_ascii", "zmin": zmin, "zmax": zmax},
    )


def read_geojson(path: str) -> PointData:
    """Read a GeoJSON file into a PointData object.

    Extracts point coordinates and properties from Feature collections.
    Also handles bare geometry collections and single geometries.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    features = []
    if data.get("type") == "FeatureCollection":
        features = data["features"]
    elif data.get("type") == "Feature":
        features = [data]
    elif data.get("type") in ("Point", "MultiPoint"):
        features = [{"type": "Feature", "geometry": data, "properties": {}}]

    coords = []
    all_props: dict[str, list] = {}

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}

        if geom.get("type") == "Point":
            c = geom["coordinates"]
            coords.append([c[0], c[1], c[2] if len(c) > 2 else 0.0])
        elif geom.get("type") == "MultiPoint":
            for c in geom["coordinates"]:
                coords.append([c[0], c[1], c[2] if len(c) > 2 else 0.0])
                for k, v in props.items():
                    all_props.setdefault(k, []).append(v)
            continue

        for k, v in props.items():
            all_props.setdefault(k, []).append(v)

    locations = np.array(coords) if coords else np.empty((0, 3))

    attributes = {}
    for k, v in all_props.items():
        try:
            attributes[k] = np.array(v, dtype=np.float64)
        except (ValueError, TypeError):
            attributes[k] = np.array(v, dtype=object)

    col_names = list(all_props.keys())

    return PointData(
        locations=locations,
        column_names=col_names,
        attributes=attributes,
        source_path=str(Path(path).resolve()),
        metadata={"format": "geojson", "n_features": len(features)},
    )


def read_csv(
    path: str,
    x_col: int | str = 0,
    y_col: int | str = 1,
    z_col: int | str | None = 2,
    value_cols: list[int | str] | None = None,
    delimiter: str = ",",
    skip_header: int = 0,
) -> PointData:
    """Read a CSV/text file with station coordinates and data.

    Args:
        path: Path to the file.
        x_col, y_col, z_col: Column index or name for coordinates.
        value_cols: Additional columns to read as data values.
        delimiter: Column separator.
        skip_header: Number of header lines to skip.
    """
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for _ in range(skip_header):
            next(f)

        reader = csv.reader(f, delimiter=delimiter)
        header = None
        rows = []

        for i, row in enumerate(reader):
            row = [c.strip() for c in row if c.strip()]
            if not row:
                continue
            try:
                float(row[0])
                rows.append(row)
            except ValueError:
                if i == 0:
                    header = row
                continue

    if not rows:
        return PointData(
            locations=np.empty((0, 3)),
            source_path=str(p.resolve()),
        )

    def _col_idx(col, hdr):
        if isinstance(col, int):
            return col
        if hdr and col in hdr:
            return hdr.index(col)
        raise ValueError(f"Column '{col}' not found in header: {hdr}")

    xi = _col_idx(x_col, header)
    yi = _col_idx(y_col, header)
    zi = _col_idx(z_col, header) if z_col is not None else None

    coords = []
    for row in rows:
        x = float(row[xi])
        y = float(row[yi])
        z = float(row[zi]) if zi is not None and zi < len(row) else 0.0
        coords.append([x, y, z])

    locations = np.array(coords)

    attributes = {}
    col_names = []
    if value_cols:
        for vc in value_cols:
            idx = _col_idx(vc, header)
            name = header[idx] if header else f"col_{idx}"
            col_names.append(name)
            vals = []
            for row in rows:
                vals.append(float(row[idx]) if idx < len(row) else np.nan)
            attributes[name] = np.array(vals)

    return PointData(
        locations=locations,
        column_names=col_names,
        attributes=attributes,
        source_path=str(p.resolve()),
        metadata={
            "format": "csv",
            "delimiter": delimiter,
            "n_rows": len(rows),
            "header": header,
        },
    )


def read_esri_ascii(path: str) -> GridData:
    """Read an ESRI ASCII grid (.asc), e.g. a DEM.

    Rows are stored north to south, so ``y`` is descending (as for GeoTIFF).
    """
    p = Path(path)
    header: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        lines = f.read().splitlines()
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 2 and not _is_number(parts[0]):
            header[parts[0].lower()] = parts[1]
            idx += 1
        else:
            break
    try:
        nx, ny = int(header["ncols"]), int(header["nrows"])
    except KeyError as e:
        raise ValueError(f"ESRI ASCII grid '{p.name}' is missing header {e}") from None
    if "cellsize" in header:
        dx = dy = float(header["cellsize"])
    else:
        dx, dy = float(header["dx"]), float(header["dy"])

    if "xllcenter" in header:
        x0 = float(header["xllcenter"])
    else:
        x0 = float(header["xllcorner"]) + dx / 2
    if "yllcenter" in header:
        y0 = float(header["yllcenter"])
    else:
        y0 = float(header["yllcorner"]) + dy / 2

    values = np.array(" ".join(lines[idx:]).split(), dtype=np.float64)
    if values.size != nx * ny:
        raise ValueError(
            f"ESRI ASCII grid '{p.name}' has {values.size} values, expected {nx * ny}"
        )
    data = values.reshape(ny, nx)
    nodata = float(header["nodata_value"]) if "nodata_value" in header else None
    if nodata is not None:
        data[data == nodata] = np.nan

    x = x0 + dx * np.arange(nx)
    y = (y0 + dy * np.arange(ny))[::-1]
    return GridData(
        values=data, x=x, y=y,
        nx=nx, ny=ny,
        xmin=x0 - dx / 2, xmax=x0 + dx * (nx - 0.5),
        ymin=y0 - dy / 2, ymax=y0 + dy * (ny - 0.5),
        dx=dx, dy=dy,
        nodata=nodata,
        source_path=str(p.resolve()),
        metadata={"format": "esri_ascii"},
    )


def read_ubc_obs(path: str) -> PointData:
    """Read a UBC-GIF gravity or magnetic observation file (.obs).

    Gravity:   ``ndata`` then rows of ``x y z value [std]``.
    Magnetics: ``inc dec geomag`` / ``a_inc a_dec dir`` / ``ndata`` then
               rows of ``x y z value [std]``.

    For magnetic files the inducing field is returned in
    ``metadata["inducing_field"]`` as (amplitude_nT, inclination, declination).
    """
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        rows = [line.split("!")[0].split() for line in f]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError(f"UBC observation file '{p.name}' is empty")

    metadata: dict[str, Any] = {"format": "ubc_obs"}
    if len(rows[0]) == 1:
        start = 1
        metadata["kind"] = "gravity"
    elif len(rows[0]) >= 3 and len(rows) > 2 and len(rows[2]) == 1:
        inc, dec, amp = (float(v) for v in rows[0][:3])
        metadata["kind"] = "magnetics"
        metadata["inducing_field"] = (amp, inc, dec)
        start = 3
    else:
        raise ValueError(f"'{p.name}' does not look like a UBC-GIF .obs file")

    data = [[float(v) for v in r[:5]] for r in rows[start:] if len(r) >= 4]
    if not data:
        raise ValueError(f"UBC observation file '{p.name}' contains no data rows")
    width = min(len(r) for r in data)
    arr = np.array([r[:width] for r in data], dtype=np.float64)

    attributes = {"std": arr[:, 4]} if width >= 5 else {}
    return PointData(
        locations=arr[:, :3],
        values=arr[:, 3],
        column_names=["x", "y", "z", "value"] + (["std"] if width >= 5 else []),
        attributes=attributes,
        source_path=str(p.resolve()),
        metadata=metadata,
    )


_X_NAMES = ("x", "easting", "east", "e", "utm_e", "utme", "utmx", "x_m", "lon", "long",
            "longitude")
_Y_NAMES = ("y", "northing", "north", "n", "utm_n", "utmn", "utmy", "y_m", "lat",
            "latitude")
_Z_NAMES = ("z", "elev", "elevation", "height", "alt", "altitude", "h", "topo", "dem",
            "z_m", "elev_m", "zobs")
_STD_NAMES = ("std", "stdev", "stddev", "sd", "sigma", "uncert", "uncertainty", "error",
              "err", "unc")
_VALUE_NAMES = ("value", "data", "obs", "observed", "d", "anomaly", "gz", "gzz", "tmi",
                "bz", "grav", "gravity", "mag", "magnetic", "magnetics", "bouguer", "cba",
                "tf", "tfa", "residual", "mgal", "nt")


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_station_table(path: str, value_name: str | None = None) -> PointData:
    """Read a tabular station file (x, y, [z,] value) with auto-detected columns.

    Columns are matched by header name when a header is present (a header
    may also be the last ``/`` or ``#`` comment line, as in Geosoft XYZ).
    Without a header: 3 columns are ``x y value``; 4 or more are
    ``x y z value``.

    Args:
        path: .csv / .txt / .dat / .xyz file.
        value_name: Preferred data column name (e.g. the component "gzz").

    Returns:
        PointData whose ``values`` is the data column (None if the file has
        only coordinates) and whose ``locations[:, 2]`` is NaN when the file
        gives no elevation.
    """
    p = Path(path)
    header: list[str] | None = None
    last_comment: list[str] | None = None
    rows: list[list[float]] = []
    delimiter: str | None = None

    with open(p, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line[0] in "#/!%":
                last_comment = line.lstrip("#/!% ").replace(",", " ").split()
                continue
            if delimiter is None:
                delimiter = "," if "," in line else (";" if ";" in line else None)
            tokens = [t.strip().strip('"').strip("'") for t in line.split(delimiter)]
            tokens = [t for t in tokens if t]
            if not rows and header is None and not _is_number(tokens[0]):
                header = tokens
                continue
            if not all(_is_number(t) for t in tokens):
                continue  # e.g. Geosoft "Line 10" separators
            rows.append([float(t) for t in tokens])

    if not rows:
        raise ValueError(f"No numeric rows found in '{p.name}'")
    width = min(len(r) for r in rows)
    if width < 3:
        raise ValueError(f"'{p.name}' needs at least 3 columns (x, y, value)")
    table = np.array([r[:width] for r in rows], dtype=np.float64)

    if header is None and last_comment and len(last_comment) >= width:
        header = last_comment
    names = [h.lower() for h in header[:width]] if header and len(header) >= width else None

    def find(cands, exclude=()):
        if names is None:
            return None
        for c in cands:
            if c in names and names.index(c) not in exclude:
                return names.index(c)
        return None

    if names is not None:
        xi, yi = find(_X_NAMES), find(_Y_NAMES)
        if xi is None or yi is None:
            xi, yi = 0, 1
        zi = find(_Z_NAMES, exclude=(xi, yi))
        used = {xi, yi} | ({zi} if zi is not None else set())
        si = find(_STD_NAMES, exclude=used)
        prefs = ((value_name.lower(),) if value_name else ()) + _VALUE_NAMES
        vi = find(prefs, exclude=used | ({si} if si is not None else set()))
        if vi is None:
            rest = [i for i in range(width) if i not in used and i != si]
            vi = rest[0] if rest else None
        col_names = header[:width]
    else:
        xi, yi = 0, 1
        zi, vi = (None, 2) if width == 3 else (2, 3)
        col_names = []

    z = table[:, zi] if zi is not None else np.full(len(table), np.nan)
    return PointData(
        locations=np.column_stack([table[:, xi], table[:, yi], z]),
        values=table[:, vi] if vi is not None else None,
        column_names=list(col_names),
        source_path=str(p.resolve()),
        metadata={
            "format": "table",
            "has_z": zi is not None,
            "columns": {"x": xi, "y": yi, "z": zi, "value": vi},
        },
    )


def read_npz_points(path: str) -> PointData:
    """Read station data from a .npz archive.

    Recognised keys: ``locations`` (or ``xyz``/``locs``), or separate
    ``x``, ``y`` [``z``]; data in ``values`` / ``observed`` / ``data`` / ``dobs``.
    """
    p = Path(path)
    with np.load(p) as npz:
        keys = set(npz.files)
        loc_key = next((k for k in ("locations", "xyz", "locs", "receiver_locations")
                        if k in keys), None)
        if loc_key is not None:
            locs = np.atleast_2d(np.asarray(npz[loc_key], dtype=np.float64))
        elif {"x", "y"} <= keys:
            cols = [np.ravel(npz["x"]), np.ravel(npz["y"])]
            if "z" in keys:
                cols.append(np.ravel(npz["z"]))
            locs = np.column_stack(cols).astype(np.float64)
        else:
            raise ValueError(
                f"'{p.name}': no 'locations' or 'x'/'y' arrays (keys: {sorted(keys)})"
            )
        val_key = next((k for k in ("values", "observed", "data", "dobs", "d") if k in keys),
                       None)
        values = np.ravel(npz[val_key]).astype(np.float64) if val_key else None

    has_z = locs.shape[1] >= 3
    if not has_z:
        locs = np.column_stack([locs[:, :2], np.full(len(locs), np.nan)])
    return PointData(
        locations=locs[:, :3],
        values=values,
        source_path=str(p.resolve()),
        metadata={"format": "npz", "has_z": has_z},
    )


def detect_format(path: str) -> str:
    """Detect file format from extension."""
    ext = Path(path).suffix.lower()
    formats = {
        ".tif": "geotiff",
        ".tiff": "geotiff",
        ".grd": "surfer_grd",
        ".asc": "esri_ascii",
        ".obs": "ubc_obs",
        ".geojson": "geojson",
        ".json": "geojson",
        ".csv": "csv",
        ".txt": "csv",
        ".dat": "csv",
        ".xyz": "csv",
        ".npy": "npy",
        ".npz": "npz",
    }
    return formats.get(ext, "unknown")


def read_auto(path: str, **kwargs) -> GridData | PointData:
    """Auto-detect format and read the file.

    Returns GridData for raster formats, PointData for vector/tabular.
    """
    fmt = detect_format(path)
    if fmt == "geotiff":
        return read_geotiff(path, **kwargs)
    elif fmt == "surfer_grd":
        return read_surfer_grd(path)
    elif fmt == "esri_ascii":
        return read_esri_ascii(path)
    elif fmt == "ubc_obs":
        return read_ubc_obs(path)
    elif fmt == "npz":
        return read_npz_points(path)
    elif fmt == "geojson":
        return read_geojson(path)
    elif fmt == "csv":
        return read_csv(path, **kwargs)
    elif fmt == "npy":
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return PointData(
                locations=arr[:, :3],
                values=arr[:, 3] if arr.shape[1] > 3 else None,
                source_path=str(Path(path).resolve()),
                metadata={"format": "npy"},
            )
        return GridData(
            values=arr,
            x=np.arange(arr.shape[-1], dtype=float),
            y=np.arange(arr.shape[0], dtype=float) if arr.ndim == 2 else np.array([0.0]),
            nx=arr.shape[-1],
            ny=arr.shape[0] if arr.ndim == 2 else 1,
            xmin=0, xmax=float(arr.shape[-1] - 1),
            ymin=0, ymax=float(arr.shape[0] - 1) if arr.ndim == 2 else 0,
            dx=1.0, dy=1.0,
            source_path=str(Path(path).resolve()),
            metadata={"format": "npy"},
        )
    else:
        raise ValueError(f"Unknown format for '{path}' (extension: {Path(path).suffix})")
