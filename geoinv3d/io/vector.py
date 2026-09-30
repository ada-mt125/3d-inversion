"""Vector data without GIS libraries: ESRI shapefiles (points, polygons) and GeoJSON.

Enough for geological constraints: borehole collars, sample points and outcrop or
interpreted-body polygons.  A shapefile is read from its .shp (geometry), .dbf
(attributes, in the .cpg encoding, UTF-8 by default) and .prj (whether it is in
longitude/latitude).  Features come back as dicts:
``{"type": "Point" | "Polygon", "coordinates": ..., "properties": {...}}`` with polygon
coordinates as a list of rings of (x, y).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

_POINT_TYPES = (1, 11, 21)
_POLY_TYPES = (5, 15, 25)
_LINE_TYPES = (3, 13, 23)


def read_dbf(path, encoding: str = "utf-8") -> list[dict]:
    """Records of a dBASE table as dicts (numbers as float, blanks as None)."""
    raw = Path(path).read_bytes()
    n_rec, header_len, rec_len = struct.unpack("<IHH", raw[4:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\0")[0].decode("latin1").strip()
        fields.append((name, chr(raw[pos + 11]), raw[pos + 16]))
        pos += 32
    out = []
    for i in range(n_rec):
        rec = raw[header_len + i * rec_len: header_len + (i + 1) * rec_len]
        if rec[:1] == b"*":   # deleted
            continue
        row, off = {}, 1
        for name, ftype, length in fields:
            text = rec[off:off + length].decode(encoding, "replace").strip()
            off += length
            if ftype in "NF":
                try:
                    row[name] = float(text) if text and not text.startswith("*") else None
                except ValueError:
                    row[name] = None
            else:
                row[name] = text or None
        out.append(row)
    return out


def _shp_features(path) -> list:
    raw = Path(path).read_bytes()
    if struct.unpack(">i", raw[:4])[0] != 9994:
        raise ValueError(f"'{Path(path).name}' is not a shapefile")
    out, pos = [], 100
    while pos + 8 <= len(raw):
        _, content_len = struct.unpack(">2i", raw[pos:pos + 8])
        body = raw[pos + 8: pos + 8 + 2 * content_len]
        pos += 8 + 2 * content_len
        shape = struct.unpack("<i", body[:4])[0]
        if shape in _POINT_TYPES:
            out.append({"type": "Point", "coordinates": struct.unpack("<2d", body[4:20])})
        elif shape in _POLY_TYPES or shape in _LINE_TYPES:
            n_parts, n_points = struct.unpack("<2i", body[36:44])
            parts = list(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])) + [n_points]
            pts_at = 44 + 4 * n_parts
            xy = struct.unpack(f"<{2 * n_points}d", body[pts_at:pts_at + 16 * n_points])
            rings = [[(xy[2 * k], xy[2 * k + 1]) for k in range(parts[j], parts[j + 1])]
                     for j in range(n_parts)]
            out.append({"type": "Polygon" if shape in _POLY_TYPES else "LineString",
                        "coordinates": rings})
        else:   # null or unsupported shape: keep the place for the attributes
            out.append({"type": None, "coordinates": None})
    return out


def read_shapefile(path) -> tuple[list[dict], bool]:
    """(features, geographic) of a shapefile given by any of its files' paths."""
    base = Path(path).with_suffix("")
    feats = _shp_features(base.with_suffix(".shp"))
    cpg = base.with_suffix(".cpg")
    enc = cpg.read_text().strip() if cpg.exists() else "utf-8"
    dbf = base.with_suffix(".dbf")
    props = read_dbf(dbf, enc if enc.lower() not in ("", "none") else "utf-8") if dbf.exists() \
        else [{} for _ in feats]
    for f, p in zip(feats, props):
        f["properties"] = p
    prj = base.with_suffix(".prj")
    geographic = prj.exists() and prj.read_text().lstrip().upper().startswith("GEOGCS")
    return feats, geographic


def read_geojson(path) -> tuple[list[dict], bool]:
    """(features, geographic) of a GeoJSON FeatureCollection; GeoJSON is lon/lat unless
    it names another CRS."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crs = (((data.get("crs") or {}).get("properties") or {}).get("name") or "")
    geographic = crs == "" or "4326" in crs or "CRS84" in crs
    out = []
    for f in data.get("features", []):
        g = f.get("geometry") or {}
        kind, coords = g.get("type"), g.get("coordinates")
        props = f.get("properties") or {}
        if kind == "Point":
            out.append({"type": "Point", "coordinates": tuple(coords[:2]), "properties": props})
        elif kind == "Polygon":
            out.append({"type": "Polygon", "coordinates": [[tuple(p[:2]) for p in r] for r in coords],
                        "properties": props})
        elif kind == "MultiPolygon":
            for poly in coords:
                out.append({"type": "Polygon",
                            "coordinates": [[tuple(p[:2]) for p in r] for r in poly],
                            "properties": props})
    return out, geographic


def read_vector(path) -> tuple[list[dict], bool]:
    """Features of a shapefile (.shp/.dbf/.shx) or GeoJSON file, and whether they are lon/lat."""
    ext = Path(path).suffix.lower()
    if ext in (".shp", ".dbf", ".shx", ".prj"):
        return read_shapefile(path)
    if ext in (".geojson", ".json"):
        return read_geojson(path)
    raise ValueError(f"Unsupported vector file '{Path(path).name}' (use a shapefile or GeoJSON)")
