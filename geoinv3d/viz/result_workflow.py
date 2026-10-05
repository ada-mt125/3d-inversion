"""Turn cloud job results into a workflow the DAG viewer can open.

A result is the ``result.zip`` of a data-pipeline job (result.json,
recovered_model.npy, active_cells.npy, data.npz).  ``build_workflow`` makes
a DAG: mesh, survey, optional true model, and one inversion node per result
with the convergence history, the 3D model (mesh core only; octree models are
sampled onto a regular grid) and the data fit.

Several runs are arranged as a tree with one setting per column: mesh and
data, regularization, α_s (the L1 share for L1–L2), depth weighting, norms
(sparse only), then the runs.  Every run passes through every column that
applies to it, so the runs under a node share every setting to its left.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import math

import numpy as np

UNITS = {"gravity": ("density", "g/cc", "mGal"), "magnetics": ("susceptibility", "SI", "nT"),
         "dc_resistivity": ("log conductivity", "ln(S/m)", "V"),
         "mt": ("log conductivity", "ln(S/m)", "Ω")}
# the data unit of a DC dataset follows its data type
DC_DATA_UNITS = {"volt": "V", "apparent_resistivity": "Ω·m"}


def _round(a, digits=5):
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=float).ravel()]


# ---------------------------------------------------------------- branch levels

REG_LABELS = {"sparse": "sparse (IRLS)", "l1l2": "L1–L2", "l2": "smooth L2", "smooth": "smooth L2",
              "mgs": "MGS focusing", "tv": "total variation",
              "group_lasso": "joint group lasso + L2", "pgi": "PGI (rock units + smoothness)"}
REG_ORDER = ["sparse", "l1l2", "l2", "smooth", "mgs", "tv", "group_lasso", "pgi"]
# how other settings are named in the tree
SETTING_LABELS = {"gl_lambda1_selection": "λ1 by", "gl_lambda1": "λ1", "gl_lambda1_ratio": "λ1/λ1max",
                  "gl_mu": "μ", "gl_data_scaling": "data scaling", "gl_gamma": "γ",
                  "gl_n_lambda1": "λ1 points", "gl_lambda1_decades": "λ1 decades",
                  "gl_max_iter": "ADMM max it", "gl_tol": "ADMM tol", "geology": "geology",
                  "coupling_weight": "coupling weight", "gl_cross_gradient": "λ3",
                  "magnetization": "magnetization", "gl_data_weights": "data weights from",
                  "gl_balance": "balanced", "gl_coupling": "group lasso pairing",
                  "bounds_lower": "lower bound", "bounds_upper": "upper bound", "max_iter": "max iterations",
                  "max_irls_iterations": "IRLS iterations", "beta_selection": "β selection",
                  "beta0_ratio": "β0 ratio", "cooling_factor": "β cooling", "alpha_x": "α_x", "alpha_y": "α_y",
                  "alpha_z": "α_z", "gl_weighting": "cell weighting", "gl_relaxation": "ADMM relaxation"}


def _g(v) -> str:
    """Short number: 1, 0.5, 1e-4."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return str(v)
    if v and abs(v) < 1e-3:
        mantissa, exponent = f"{v:e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return f"{v:g}"


def _sortable(v):
    """A sort key for setting values that may mix None, numbers and strings."""
    if isinstance(v, (list, tuple)):
        return (3, tuple(_sortable(x) for x in v))
    if v is None:
        return (0, 0)
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def _norms(p):
    return "[" + ", ".join(_g(x) for x in p) + "]"


# Each level maps a run's settings to (sort key, node label, short form for the result card).
def _coupling_level(s):
    """How a joint run's models are coupled (methods/coupling.py); single runs: none."""
    from ..methods.coupling import coupling_label
    c = s.get("coupling")
    if not c:
        return (0, ""), "single inversion", "single"
    label, w = coupling_label(c), s.get("coupling_weight")
    weight = f" · w = {_g(w)}" if w is not None else ""   # the weight, where the coupling has one
    return (1, c, w if w is not None else 0), f"coupling: {label}{weight}", f"{label}{weight}"


def _reg_level(s):
    rt = s.get("regularization_type")
    label = REG_LABELS.get(rt, str(rt))
    return (REG_ORDER.index(rt) if rt in REG_ORDER else len(REG_ORDER), rt), label, label


def _strength_level(s):
    """α_s, the L1 share of the L1–L2 elastic net, or the L2 damping λ2 of the group
    lasso: the weight of the model-norm term."""
    if s.get("regularization_type") == "l1l2":
        r = s.get("l1_ratio")
        return (1, r), f"L1 share {_g(r)}", f"L1 share {_g(r)}"
    if s.get("regularization_type") == "group_lasso":
        v = s.get("gl_lambda2")
        return (2, v), f"λ2 = {_g(v)}", f"λ2={_g(v)}"
    a = s.get("alpha_s")
    return (0, a), f"α_s = {_g(a)}", f"α_s={_g(a)}"


def _weighting_level(s):
    if s.get("depth_weighting") == "depth":
        b = s.get("depth_weighting_exponent")
        return (1, b), f"depth, β = {_g(b)}", f"β={_g(b)}"
    if s.get("depth_weighting") == "sensitivity_volume":   # the group lasso's
        return (0, 1), "sensitivity × cell volume", "sensitivity×volume"
    if s.get("depth_weighting") == "per model":
        return (2, 0), "per model", "per model"
    return (0, 0), "sensitivity", "sensitivity"


def _norms_level(s):
    p = list(s.get("norms") or [])
    return p, f"p = {_norms(p)}", f"p={_norms(p).replace(' ', '')}"


def _setting_level(key):
    name = SETTING_LABELS.get(key, key)

    def level(s):
        v = s.get(key)
        shown = "none" if v is None else _g(v)
        return v, f"{name} = {shown}", f"{name}={shown}"
    return level


# The columns of a study's tree, in order: (name, regularization types, level).  Every run
# passes through every level that applies to its regularization, so a column always holds
# one setting and the runs under a node differ only in the columns to its right.
BRANCH_LEVELS = [
    ("regularization", None, _reg_level),
    ("strength", None, _strength_level),
    ("weighting", None, _weighting_level),
    ("norms", {"sparse"}, _norms_level),
]
# settings the levels above cover, and settings a regularization ignores (still recorded)
COVERED = {"regularization_type", "alpha_s", "l1_ratio", "depth_weighting", "depth_weighting_exponent", "norms",
           "gl_lambda2", "coupling", "coupling_weight", "gl_weighting"}
IGNORED = {"l1l2": {"alpha_s", "norms", "alpha_x", "alpha_y", "alpha_z", "max_irls_iterations"},
           "l2": {"norms", "max_irls_iterations"}, "smooth": {"norms", "max_irls_iterations"},
           # PGI regularizes by its rock units: none of the per-model settings apply
           "pgi": {"alpha_s", "norms", "l1_ratio", "max_irls_iterations", "depth_weighting",
                   "depth_weighting_exponent", "coupling_weight"}}


LEVEL_TITLES = {"coupling": "Coupling", "regularization": "Regularization", "weighting": "Depth weighting",
                "norms": "Norms p"}


def _level_title(name, s):
    """The title of a tree column: the setting its nodes fix (shown over the column)."""
    if name == "strength":
        return {"l1l2": "L1 share", "group_lasso": "λ2"}.get(s.get("regularization_type"), "α_s")
    if name in LEVEL_TITLES:
        return LEVEL_TITLES[name]
    t = SETTING_LABELS.get(name, name.replace("_", " "))
    return t[:1].upper() + t[1:]


def _levels(rt, extra_keys):
    """The levels of runs with regularization ``rt`` (after the regularization itself):
    the fixed ones, then the other settings that differ within the study."""
    levels = [] if rt == "pgi" else [(name, fn) for name, types, fn in BRANCH_LEVELS[1:]
                                     if types is None or rt in types]
    ignored = IGNORED.get(rt, set())
    return levels + [(k, _setting_level(k)) for k in extra_keys if k not in ignored]


def _extra_keys(runs):
    """Settings other than the fixed levels that differ between runs of the same regularization."""
    keys = set()
    by_type = {}
    for r in runs:
        s = r.get("settings", {})
        by_type.setdefault(s.get("regularization_type") or r.get("regularization"), []).append(s)
    for rt, ss in by_type.items():
        for k in {k for s in ss for k in s} - COVERED - IGNORED.get(rt, set()):
            if len({json.dumps(s.get(k)) for s in ss}) > 1:
                keys.add(k)
    return sorted(keys)


def load_result(path) -> dict:
    """result.json plus the arrays of a result.zip (or of a directory holding one).

    Arrays go under ``_model``, ``_active`` and ``_data`` (stations, observed,
    std, predicted); ``_name`` is the job's task or file name.  A joint run
    has ``_models`` and ``_datas`` instead, keyed by method.
    """
    path = Path(path)
    zpath = path / "result.zip" if path.is_dir() else path
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        meta = json.loads(zf.read("result.json"))
        read = lambda n: np.load(io.BytesIO(zf.read(n)))   # noqa: E731

        def read_npz(n):
            with np.load(io.BytesIO(zf.read(n))) as d:
                return {k: d[k] for k in d.files}
        meta["_model"] = read("recovered_model.npy") if "recovered_model.npy" in names else None
        meta["_active"] = read("active_cells.npy") if "active_cells.npy" in names else None
        if "data.npz" in names:
            meta["_data"] = read_npz("data.npz")
        models = {n[len("recovered_"):-len(".npy")]: read(n) for n in names
                  if n.startswith("recovered_") and n.endswith(".npy") and n != "recovered_model.npy"}
        datas = {n[len("data_"):-len(".npz")]: read_npz(n) for n in names
                 if n.startswith("data_") and n.endswith(".npz")}
        if models:
            meta["_models"] = models
        if datas:
            meta["_datas"] = datas
        if "topography.npz" in names:
            meta["_topography"] = read_npz("topography.npz")
        if "reference_model.npy" in names:   # geology constraints, on the active cells
            meta["_reference"] = read("reference_model.npy")
        for key in ("posterior_std", "prob_body", "prior_std", "samples"):   # a Bayesian posterior
            if f"{key}.npy" in names:
                meta[f"_{key}"] = read(f"{key}.npy")
    meta["_name"] = path.name if path.is_dir() else path.stem.replace("_result", "")
    return meta


def split_joint_runs(runs: list[dict]) -> tuple[list[dict], list[str]]:
    """One run per property model of each joint run, so that the viewer shows it
    like a single inversion (its own data node, 3D model and data fit).

    A model is shown with the data of its first dataset (``dataset_models``
    maps datasets to models; without it, a model's dataset has its name).
    Joint runs without per-dataset data (e.g. legacy cross-gradient results)
    cannot be shown: their names come back as the second value.
    """
    out, skipped = [], []
    for run in runs:
        if run.get("_model") is not None or not run.get("_models"):
            out.append(run)
            continue
        models, datas = run["_models"], run.get("_datas") or {}
        labels = run.get("dataset_labels") or []
        of_model = dict(zip(labels, run.get("dataset_models") or labels))
        # each model with the first of its datasets that has data
        parts = []
        for name in sorted(models):
            mine = [lbl for lbl in labels if of_model.get(lbl) == name and lbl in datas]
            if name in datas and name not in mine:
                mine.insert(0, name)
            if mine:
                parts.append((name, mine[0]))
        if not parts:
            skipped.append(str(run.get("_name")))
            continue
        runs_datasets = run.get("datasets", [])
        for name, label in parts:
            k = labels.index(label) if label in labels else None
            ds = runs_datasets[k] if k is not None and k < len(runs_datasets) else next(
                (d for d in runs_datasets if d.get("method") == label), {"method": label})
            prop = UNITS.get(ds.get("method"), UNITS.get(name, (name,)))[0]
            part = {**run, "_model": models[name], "_data": datas[label], "datasets": [ds],
                    "method": ds.get("method", name), "_joint_part": prop,
                    "_name": f"{run.get('_name')} · {name if name not in UNITS else prop}"}
            # two models of one run: each is shown with the other as an overlay in the 3D view
            others = [n for n, _ in parts if n != name]
            if len(parts) == 2 and not run.get("_overlay"):
                o_prop, o_unit, _ = UNITS.get(others[0], (others[0], "", ""))
                part["_overlay"] = {"model": models[others[0]], "label": f"{o_prop.capitalize()} (same run)",
                                    "unit": o_unit}
            out.append(part)
    return out, skipped


def result_mesh(meta):
    import copy
    import discretize
    # deserialize() pops keys from the dict it is given, so hand it a copy
    spec = copy.deepcopy(meta["mesh"])
    return getattr(discretize, spec["__class__"]).deserialize(spec)


def full_model(meta, mesh) -> np.ndarray:
    """Model on every mesh cell (inactive cells, i.e. air, set to 0)."""
    m = np.zeros(mesh.n_cells)
    if meta["_active"] is not None:
        m[meta["_active"]] = meta["_model"]
    else:
        m[:] = meta["_model"]
    return m


MAX_VIEWER_CELLS = 600_000   # voxels of the 3D tab: Plotly's isosurfaces crawl beyond


def _coarsening(n_xy, n_z, h, dz, max_cells=None):
    """Integer factors (k, kz) for the viewer's horizontal and vertical cells that keep
    (nx/k)(ny/k)(nz/kz) within ``max_cells``: the finer of the two sizes is doubled up
    first, so the voxels stay about as wide as they are tall."""
    nx, ny = n_xy
    max_cells = max_cells or MAX_VIEWER_CELLS
    k = kz = 1
    while math.ceil(nx / k) * math.ceil(ny / k) * math.ceil(n_z / kz) > max_cells:
        if h * k <= dz * kz and (nx > k or ny > k):
            k += 1
        elif n_z > kz:
            kz += 1
        else:
            k += 1
    return k, kz


def _block_mean(v, k, kz):
    """Means of k x k x kz blocks of a 3-D array (partial blocks at the far ends)."""
    if k == 1 and kz == 1:
        return v
    nx, ny, nz = v.shape
    px, py, pz = -nx % k, -ny % k, -nz % kz
    w = np.pad(v.astype(float), ((0, px), (0, py), (0, pz)), constant_values=np.nan)
    w = w.reshape((nx + px) // k, k, (ny + py) // k, k, (nz + pz) // kz, kz)
    with np.errstate(invalid="ignore"):
        return np.nanmean(w, axis=(1, 3, 5))


def _coarse_edges(edges, k):
    """Every k-th edge, with the last one kept."""
    e = np.asarray(edges)
    out = e[::k]
    return out if out[-1] == e[-1] else np.append(out, e[-1])


class ViewerGrid:
    """The regular grid the viewer's 3D tab shows: the core of the mesh.

    Tensor meshes keep their core cells (padding dropped): below a DEM these
    include the layers the relief adds on top of the core depth, all of the core
    thickness.  Octree meshes are sampled at the centres of a grid of core-sized
    cells from the highest ground over the stations (not the top of the octree's box,
    which reaches far into the air) down to the core depth below the lowest ground.
    Cells above the ground are air (0 in the model).  Grids of more than
    MAX_VIEWER_CELLS are coarsened by whole cells (tensor: block means; octree: sampled
    on the coarser grid).
    """

    def __init__(self, meta, mesh):
        from ..cloud.meshing import padding_cells
        used = meta["mesh_design"]["used"]
        self.mesh = mesh
        h, dz = used["core_cell_m"], used["core_cell_z_m"]
        n_core_z = int(round(used["depth_core_m"] / dz))
        self.factors = (1, 1)
        from discretize import TensorMesh
        if isinstance(mesh, TensorMesh):
            n_pad = padding_cells(h, used["pad_distance_m"])
            nx, ny, nz = mesh.shape_cells
            hz = np.asarray(mesh.h[2])
            uniform = np.abs(hz - dz) <= 1e-6 * dz
            n_top = int(np.argmin(uniform[::-1])) if not uniform.all() else nz
            n_z = max(n_core_z, n_top)   # the core plus the relief layers
            self.sl = (slice(n_pad, nx - n_pad), slice(n_pad, ny - n_pad),
                       slice(max(nz - n_z, 0), nz))
            x_edges = mesh.nodes_x[self.sl[0].start:self.sl[0].stop + 1]
            y_edges = mesh.nodes_y[self.sl[1].start:self.sl[1].stop + 1]
            z_edges = mesh.nodes_z[self.sl[2].start:self.sl[2].stop + 1]
            k, kz = self.factors = _coarsening((len(x_edges) - 1, len(y_edges) - 1),
                                               len(z_edges) - 1, h, dz)
            self.x_edges, self.y_edges = _coarse_edges(x_edges, k), _coarse_edges(y_edges, k)
            # z blocks are counted from the top, where the ground is
            self.z_edges = _coarse_edges(z_edges[::-1], kz)[::-1]
            self.index = None
        else:   # octree: sample on a regular grid over the stations
            ext = meta.get("cell_centers_x", [0, 0]) + meta.get("cell_centers_y", [0, 0])
            cc, hz = mesh.cell_centers, mesh.h_gridded[:, 2]
            active = meta.get("_active")
            ground = np.ones(mesh.n_cells, bool) if active is None else np.asarray(active, bool)
            over = ground & (cc[:, 0] >= ext[0]) & (cc[:, 0] <= ext[1]) & \
                (cc[:, 1] >= ext[2]) & (cc[:, 1] <= ext[3])
            sel = over if over.any() else ground
            # the top of the highest ground cell over the stations: above it is only air
            top = float((cc[sel, 2] + 0.5 * hz[sel]).max()) if sel.any() else float(cc[:, 2].max())
            low = (meta.get("topography") or {}).get("elevation_min")
            bottom = (top if low is None else min(float(low), top)) - used["depth_core_m"]
            n_z = max(n_core_z, int(np.ceil((top - bottom) / dz - 1e-9)))
            nx = len(np.arange(ext[0], ext[1] + h, h)) - 1
            ny = len(np.arange(ext[2], ext[3] + h, h)) - 1
            k, kz = self.factors = _coarsening((nx, ny), n_z, h, dz)
            self.x_edges = np.arange(ext[0], ext[1] + h * k, h * k)
            self.y_edges = np.arange(ext[2], ext[3] + h * k, h * k)
            self.z_edges = top - dz * kz * np.arange(int(np.ceil(n_z / kz)) + 1)[::-1]
            xc = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
            yc = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
            zc = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
            pts = np.stack(np.meshgrid(xc, yc, zc, indexing="ij"), -1).reshape(-1, 3, order="F")
            self.index = mesh.get_containing_cells(pts)
            self.sl = None
        self.shape = (len(self.x_edges) - 1, len(self.y_edges) - 1, len(self.z_edges) - 1)

    def values(self, full_values) -> np.ndarray:
        """Viewer order: x fastest, then y, then z from the top down."""
        v = np.asarray(full_values)
        if self.sl is not None:
            v = v.reshape(tuple(self.mesh.shape_cells), order="F")[self.sl]
            k, kz = self.factors
            if (k, kz) != (1, 1):   # blocks from the top: flip, average, flip back
                v = _block_mean(v[:, :, ::-1], k, kz)[:, :, ::-1]
        else:
            v = v[self.index].reshape(self.shape, order="F")
        return v[:, :, ::-1].ravel(order="F")

    def geometry(self) -> dict:
        return {"nx": self.shape[0], "ny": self.shape[1], "nz": self.shape[2],
                "x_edges": _round(self.x_edges), "y_edges": _round(self.y_edges),
                "z_edges": _round(self.z_edges[::-1])}


def _mesh_key(run) -> str:
    return json.dumps([run.get("mesh_type"), run.get("mesh_shape"), run["mesh_design"]["used"]],
                      sort_keys=True)


def _data_key(run) -> tuple:
    """Runs on the same data share a Survey node and a tree; a run's ``_study`` label
    (optional) keeps the runs of different studies of the same data in trees of their own."""
    d = run["_data"]
    return (run.get("_study"), len(d["observed"]), round(float(np.sum(d["locations"][:, :2])), 3),
            round(float(np.sum(d["observed"])), 6))


def build_workflow(runs: list[dict], true_model=None, true_label: str = "True model") -> dict:
    """A viewer workflow from loaded results (see :func:`load_result`).

    Runs on the same mesh share one Mesh node, runs on the same data one
    Survey node; runs may differ in both (e.g. a coarse parameter study next
    to a full-resolution run).  ``true_model``, if given, holds a value for
    every cell of the first run's mesh; a dict {method: model} (e.g. "gravity",
    "magnetics") gives each property of a joint study its own true model.  Each run needs ``_data`` with
    stations, observed, std and predicted data; ``_backend`` (default "ec2")
    labels where it ran, ``_job_id`` (with ``_rerun``: its inputs were kept) the API job, ``_study`` (optional) the study it belongs to (see :func:`_data_key`).
    ``_overlay`` = {"model": values on the run's cells, "label", "unit"} adds a second layer
    to the run's 3D view, drawn on its own scale (another property on the same mesh); the two
    models of a joint run get each other as overlay.  A joint run with per-dataset data shows as one run per
    property model (see :func:`split_joint_runs`); one without is left out and
    named in the workflow's ``skipped``.
    """
    runs, skipped = split_joint_runs(runs)
    if not runs:
        raise ValueError("None of these results can be shown: joint results of the "
                         "cross-gradient inversion carry no per-dataset data")
    for run in runs:
        if run.get("_data") is None:
            raise ValueError("The result has no data.npz (made by an older worker): "
                             "pass the observed and predicted data explicitly")
    first = runs[0]
    method = first["datasets"][0]["method"]
    prop = UNITS.get(method, ("model", "", ""))[0]

    def units(run):
        ds = run["datasets"][0]
        prop, model_unit, data_unit = UNITS.get(ds["method"], ("model", "", ""))
        if ds["method"] == "dc_resistivity":
            data_unit = DC_DATA_UNITS.get(ds.get("component"), data_unit)
        return prop, model_unit, data_unit

    # ids: the first mesh / survey / true model keep 1, 2, 3; inversions are
    # 10 + i; further meshes, surveys and branch nodes follow the inversions
    extra_ids = iter(range(10 + len(runs), 10 ** 6))
    nodes, meshes, surveys = [], {}, {}

    def mesh_node(run):
        key = _mesh_key(run)
        if key not in meshes:
            mesh = result_mesh(run)
            node_id = 1 if not meshes else next(extra_ids)
            used = run["mesh_design"]["used"]
            nodes.append({
                "id": node_id, "type": "MeshCreateNode",
                "name": f"{run.get('mesh_type', 'tensor').title()} mesh {used['core_cell_m']:g} m",
                "inputs": [], "params": {**used, "mesh_type": run.get("mesh_type"),
                                         "source": run["mesh_design"].get("source")},
                "output": {"type": "Mesh3D", "shape": list(run.get("mesh_shape", [])),
                           "n_cells": int(mesh.n_cells)}})
            meshes[key] = (node_id, mesh, ViewerGrid(run, mesh))
        return meshes[key]

    def survey_node(run):
        key = _data_key(run)
        if key not in surveys:
            node_id = 2 if not surveys else next(extra_ids)
            d, rds = run["_data"], run["datasets"][0]
            observed = d["observed"]
            nodes.append({
                "id": node_id, "type": "SurveyCreateNode",
                "name": (f"{run['_study']} · " if run.get("_study") else "")
                + f"{rds['method']} · {len(observed)} stations", "inputs": [],
                "params": {"method": rds["method"], "component": rds.get("component"),
                           "files": rds.get("files"), "noise_pct": rds.get("noise_pct"),
                           "noise_floor": rds.get("noise_floor")},
                "output": {"type": "SurveyData", "n_stations": int(len(observed)),
                           "method": rds["method"], "data_min": float(np.min(observed)),
                           "data_max": float(np.max(observed))}})
            surveys[key] = node_id
        return surveys[key]

    first_mesh_id, first_mesh, first_grid = mesh_node(first)
    survey_node(first)
    # the true model(s): one for every run, or one per method (a joint study's properties)
    truths = true_model if isinstance(true_model, dict) else (
        {None: true_model} if true_model is not None else {})
    true_grids, true_ids = {}, {}
    for k, (key, tm) in enumerate(truths.items()):
        tm = np.asarray(tm, dtype=float)
        tprop = UNITS.get(key, (prop,))[0] if key is not None else prop
        true_grids[key] = _round(first_grid.values(tm), 4)
        true_ids[key] = 3 if k == 0 else next(extra_ids)
        nodes.append({"id": true_ids[key], "type": "ModelFromArrayNode",
                      "name": true_label if key is None else f"{true_label} · {tprop}",
                      "inputs": [first_mesh_id], "params": {"prop": tprop},
                      "output": {"type": "PhysicalModel", "prop": tprop,
                                 "n_cells": int(first_mesh.n_cells),
                                 "min": float(tm.min()), "max": float(tm.max()),
                                 "mean": float(tm.mean())}})

    def truth_key(run):
        method = run["datasets"][0]["method"]
        return method if method in truths else (None if None in truths else "")

    order = iter(range(10 ** 6))   # depth-first position, for the viewer's layout

    def settings_of(run):
        s = dict(run.get("settings", {}))
        s["regularization_type"] = s.get("regularization_type") or run.get("regularization")
        # results from before the solver was recorded: their label names it
        if s["regularization_type"] == "l1l2" and "l1l2_solver" not in s:
            label = str(run.get("regularization") or "")
            if label.endswith("_IRLS") or label.endswith("_CDA"):
                s["l1l2_solver"] = label.rsplit("_", 1)[1].lower()
        return s

    def split(group, fn):
        """The runs of ``group`` by their value of a level, sorted: [(label, short, runs)]."""
        buckets = {}
        for i, run in group:
            key, label, short = fn(settings_of(run))
            buckets.setdefault(json.dumps(_sortable(key)), (_sortable(key), label, short, []))[3].append((i, run))
        return [b[1:] for b in sorted(buckets.values(), key=lambda b: b[0])]

    def grow(group, inputs, levels, path, shorts, extra_keys, names=()):
        """One node per value of the first of ``levels``, then the next level below it;
        the runs hang from the last level (named by their levels, the regularization,
        the tree's own first split, left out — but a joint run's coupling kept)."""
        if not levels:
            for i, run in group:
                label = " · ".join(sh for n, sh in zip(names, shorts) if n != "regularization")
                inversion_node(i, run, inputs, label or shorts[0], path)
            return
        (name, fn), rest = levels[0], levels[1:]
        for label, short, sub in split(group, fn):
            node_id = next(extra_ids)
            nodes.append({
                "id": node_id, "order": next(order),
                "type": "RegularizationBranchNode" if name == "regularization" else "ParameterBranchNode",
                "name": label, "inputs": inputs, "params": {},
                "branch": {"level": name, "title": _level_title(name, settings_of(sub[0][1])), "label": label,
                           "path": path + [label], "n_runs": len(sub)}})
            below = _levels(settings_of(sub[0][1])["regularization_type"], extra_keys) if name == "regularization" else rest
            grow(sub, [node_id], below, path + [label], shorts + [short], extra_keys, names + (name,))

    def inversion_node(i, run, inputs, name, path=None):
        mesh_id, mesh, grid = mesh_node(run)
        d = run["_data"]
        locs, observed, pred = d["locations"], d["observed"], d.get("predicted")
        m_full = full_model(run, mesh)
        settings = settings_of(run)
        run_method = run["datasets"][0]["method"]
        prop, model_unit, data_unit = units(run)
        out = {
            "type": "InversionResult", "method": run_method,
            "converged": run.get("converged", True), "n_iterations": run.get("n_iterations"),
            # how the run ended against its target misfit (worker.assess_convergence)
            **({"convergence": run["convergence"]} if run.get("convergence") else {}),
            "iterations": run.get("iterations", []), "regularization": run.get("regularization"),
            "final_model": {"prop": prop, "unit": model_unit, "n_cells": int(len(run["_model"])),
                            "min": float(run["_model"].min()), "max": float(run["_model"].max()),
                            "mean": float(run["_model"].mean())},
            "model_3d": {**grid.geometry(), "values": _round(grid.values(m_full), 4)},
        }
        if run.get("_prob_body") is not None:   # a Bayesian posterior: the probability of a body
            prob_full = full_model({**run, "_model": run["_prob_body"]}, mesh)
            out["model_3d"]["prob_values"] = _round(grid.values(prob_full), 3)
        true_grid = true_grids.get(truth_key(run))
        if true_grid is not None and mesh_id == first_mesh_id:
            out["model_3d"]["true_values"] = true_grid
        elif run.get("_reference") is not None:   # the geology constraints, to compare with
            ref_full = full_model({**run, "_model": run["_reference"]}, mesh)
            out["model_3d"]["true_values"] = _round(grid.values(ref_full), 4)
            out["model_3d"]["true_label"] = "Geology reference"
        elif run.get("_overlay") and len(run["_overlay"]["model"]) == len(run["_model"]):
            # another property on the same cells (e.g. the susceptibility of a joint run over
            # its density): a second layer of the 3D view, on its own scale
            ov = run["_overlay"]
            ov_full = full_model({**run, "_model": np.asarray(ov["model"], dtype=float)}, mesh)
            out["model_3d"].update(true_values=_round(grid.values(ov_full), 4), true_other=True,
                                   true_label=ov.get("label") or "Overlay", true_unit=ov.get("unit") or "")
        topo = run.get("_topography")
        if topo is not None:   # the ground, drawn over the model in the 3D view
            out["model_3d"]["surface"] = {"x": _round(topo["x"], 7), "y": _round(topo["y"], 7),
                                          "z": [_round(row, 5) for row in np.asarray(topo["z"])]}
        if np.isfinite(locs[:, 2]).all():   # the stations at their heights
            out["model_3d"]["stations"] = {"x": _round(locs[:, 0], 7), "y": _round(locs[:, 1], 7),
                                           "z": _round(locs[:, 2], 6)}
        if pred is not None:
            resid = observed - pred
            out["data_fit"] = {
                "unit": data_unit, "x_stations": _round(locs[:, 0], 7),
                "y_stations": _round(locs[:, 1], 7), "observed": _round(observed),
                "predicted": _round(pred), "residuals": _round(resid),
                "rms": float(np.sqrt(np.mean(resid ** 2))),
                "chi2": float(np.sum((resid / d["std"]) ** 2) / len(resid))}
        params = {"method_type": run_method, "backend": run.get("_backend", "ec2"),
                  "task": run.get("_name"), "mesh_type": run.get("mesh_type"),
                  "n_data": run.get("n_data"),
                  **{k: v for k, v in settings.items() if v is not None}}
        if run.get("_joint_part"):
            params["joint_inversion"] = " + ".join(run.get("methods") or [])
        info = run.get("group_lasso")
        if info:   # what the run chose, next to what was asked
            params.update({f"gl_{k}_chosen" if k in ("lambda1", "mu") else f"gl_{k}": info.get(k)
                           for k in ("lambda1", "criterion", "mu", "n_active_cells", "solver",
                                     "admm_iterations", "admm_converged") if info.get(k) is not None})
            if info.get("chi2"):
                params["chi2_joint"] = info["chi2"]
            if info.get("warnings"):
                params["gl_warnings"] = info["warnings"]
        if run.get("notes"):
            params["notes"] = run["notes"]
        node = {"id": 10 + i, "order": next(order), "type": "RegularizedInversionNode", "name": name,
                "inputs": inputs, "params": params, "output": out}
        if run.get("_job_id"):   # a job of the API server: the page can run it again with changes
            node["job"] = {"id": run["_job_id"], "rerun": bool(run.get("_rerun")),
                           "backend": run.get("_backend", "ec2")}
        if path is not None:
            node["branch"] = {"level": "run", "label": name, "path": path}
        nodes.append(node)

    # (a joint run's two parts go to different data, hence different trees)
    # runs on the same mesh and data: one tree per pair
    groups = {}
    for i, run in enumerate(runs):
        mesh_id = mesh_node(run)[0]
        groups.setdefault((mesh_id, survey_node(run)), []).append((i, run))
    for (mesh_id, survey_id), group in groups.items():
        # the columns of this tree: the settings that differ among its own runs (settings that
        # differ only between studies, e.g. the bounds of two methods, say nothing within one)
        extra_keys = _extra_keys([run for _, run in group])
        tid = true_ids.get(truth_key(group[0][1]))
        inputs = [mesh_id, survey_id] + ([tid] if tid is not None and mesh_id == first_mesh_id else [])
        if len(runs) == 1:   # a single job: no tree
            i, run = group[0]
            inversion_node(i, run, inputs, f"{settings_of(run)['regularization_type']} · {run.get('_name')}")
            continue
        # joint runs: the coupling first (a column of its own, before the regularization)
        joint = any(settings_of(run).get("coupling") for _, run in group)
        first = len(nodes)
        grow(group, inputs, ([("coupling", _coupling_level)] if joint else [])
             + [("regularization", _reg_level)], [], [], extra_keys)
        # one column per setting in this tree, in the order of the levels: a run whose
        # regularization skips a setting (the norms of an L1–L2 run) skips its column
        rank = {"coupling": 0, "regularization": 1, **{name: 2 + i for i, (name, _, _) in enumerate(BRANCH_LEVELS[1:])},
                **{k: 2 + len(BRANCH_LEVELS) + i for i, k in enumerate(extra_keys)}}
        made = [n for n in nodes[first:] if n["type"].endswith("BranchNode")]
        columns = sorted({n["branch"]["level"] for n in made}, key=lambda name: rank.get(name, len(rank)))
        for n in made:
            n["branch"]["column"] = columns.index(n["branch"]["level"])
    nodes.sort(key=lambda n: n["id"])
    workflow = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "nodes": nodes}
    if skipped:
        workflow["skipped"] = skipped
    return workflow
