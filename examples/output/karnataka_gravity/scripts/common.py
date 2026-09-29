"""Shared paths, run registry and depth metrics for the Karnataka gravity comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent                     # examples/output/karnataka_gravity
DATA = ROOT / "data"
FIGS = ROOT / "figures"
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from geoinv3d.cloud.meshing import padding_cells  # noqa: E402
from geoinv3d.viz.result_workflow import full_model, load_result, result_mesh  # noqa: E402

AOI = (641000.0, 711000.0, 1634000.0, 1704000.0)
MAIN_HIGH = (671500.0, 1664500.0)         # the main Bouguer high
NW_HIGH = (650500.0, 1682500.0)
SPECTRAL_DEPTHS_KM = None                 # filled by make_figures (spectrum)

# Full-resolution runs on EC2 (1 km x 1 km x 500 m cells, 190,512 cells, 5,040 data)
FULL = [
    # key, zh label, en label, colour, line style
    ("original_sparse", "原设置 sparse（α_s=1e-4，灵敏度加权）", "Original sparse (α_s=1e-4, sensitivity)", "#7f7f7f", "-"),
    ("l1l2_irls", "L1–L2（IRLS）", "L1–L2 (IRLS)", "#1f5f8b", "-"),
    ("as1_beta0.5", "sparse α_s=1，β=0.5", "sparse α_s=1, β=0.5", "#2e9c6a", "--"),
    ("as1_beta1", "sparse α_s=1，β=1", "sparse α_s=1, β=1", "#2e9c6a", "-"),
    ("as1_beta1.5", "sparse α_s=1，β=1.5", "sparse α_s=1, β=1.5", "#2e9c6a", ":"),
    ("as0.1_beta1", "sparse α_s=0.1，β=1", "sparse α_s=0.1, β=1", "#d68910", "-"),
]
# 2 km study runs (local, every 4th node: 1,295 data)
LOWRES = ["base", "as1e-2", "as0.1", "as1", "as1_p0111", "as1_p0222", "as1_p0220", "as1e-4_p0222",
          "as1_p0222_dw0.5", "as1_p0222_dw1", "as1_p0222_dw1.5", "as1_p0222_dw2",
          "as1_p0221_dw1", "as1_p0111_dw1", "as0.1_p0222_dw1", "base_dw1"]


def load_full(key):
    return load_result(DATA / "ec2_runs" / key)


def load_lowres(key):
    return load_result(DATA / "lowres_runs" / key)


def grids(meta):
    """Model on the tensor mesh as (nx, ny, nz), cell volumes, depth of each layer."""
    mesh = result_mesh(meta)
    shape = tuple(mesh.shape_cells)
    m = full_model(meta, mesh).reshape(shape, order="F")
    used = meta["mesh_design"]["used"]
    n_pad = padding_cells(used["core_cell_m"], used["pad_distance_m"])
    n_core_z = int(round(used["depth_core_m"] / used["core_cell_z_m"]))
    vol = mesh.cell_volumes.reshape(shape, order="F")
    depth = -mesh.cell_centers_z
    return mesh, m, vol, depth, n_pad, n_core_z


def metrics(meta):
    """Where the anomalous mass lies: core / lateral padding / below the core, D50, D90."""
    mesh, m, vol, depth, n_pad, n_core_z = grids(meta)
    nx, ny, _ = m.shape
    a = np.abs(m * vol)
    core_xy = np.zeros((nx, ny), bool)
    core_xy[n_pad:nx - n_pad, n_pad:ny - n_pad] = True
    below = depth > n_core_z * meta["mesh_design"]["used"]["core_cell_z_m"] + 1e-6
    tot = a.sum()
    out = {"core": float(a[core_xy[:, :, None] & ~below[None, None, :]].sum() / tot),
           "lateral": float(a[~core_xy[:, :, None] & ~below[None, None, :]].sum() / tot),
           "below": float(a[np.broadcast_to(below[None, None, :], m.shape)].sum() / tot)}
    col = a[core_xy].sum(axis=0)
    o = np.argsort(depth)
    cum = np.cumsum(col[o]) / col.sum()
    out["D50"] = float(np.interp(0.5, cum, depth[o]) / 1e3)
    out["D90"] = float(np.interp(0.9, cum, depth[o]) / 1e3)
    return out


def column(meta, xy):
    """Density profile under (x, y): depth (km, ascending) and values."""
    mesh, m, _, depth, _, _ = grids(meta)
    i = int(np.argmin(np.abs(mesh.cell_centers_x - xy[0])))
    j = int(np.argmin(np.abs(mesh.cell_centers_y - xy[1])))
    o = np.argsort(depth)
    return depth[o] / 1e3, m[i, j, :][o]


def body(z, v):
    """Peak, the half-maximum depth range and the centroid depth of a profile.

    Models held at the upper bound have a flat peak, so the depth of "the"
    peak is not defined; the centroid (positive density x thickness) is.
    """
    inside = z[v >= v.max() / 2]
    w = np.clip(v, 0, None) * np.gradient(z)
    return {"peak": float(v.max()), "top_km": float(inside.min()), "bottom_km": float(inside.max()),
            "centroid_km": float((w @ z) / w.sum())}


def data_fit(meta):
    d = meta["_data"]
    r = d["observed"] - d["predicted"]
    return {"chi2": float(np.mean((r / d["std"]) ** 2)), "rms": float(np.sqrt(np.mean(r ** 2))),
            "max_abs": float(np.abs(r).max()), "corr": float(np.corrcoef(d["observed"], d["predicted"])[0, 1])}


def simpeg_8km():
    return json.loads((DATA / "simpeg_8km_mesh_profiles.json").read_text(encoding="utf-8"))
