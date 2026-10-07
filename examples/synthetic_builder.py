"""Synthetic data of a model drawn in the model builder, inverted on a coarse mesh: gravity,
magnetics, DC resistivity and MT, each without and with the model's boreholes as constraints.

The model (MODEL, in the builder's own format; OUT/model.json opens in the page, /model,
with "⤒ Import"): a 2 km x 2 km area, a 50 m conductive cover (20 ohm m) over a resistive
basement (1000 ohm m), a sulphide body dipping 60 degrees east (3.3 g/cc, 0.05 SI, 3 ohm m;
100-400 m deep) and a granite stock (2.57 g/cc, 5000 ohm m; 150-800 m deep), and two
vertical holes logged through them (HOLES).  The values on a mesh come from
geoinv3d.methods.builder.model_values (each cell the volume average of what it holds), as
the page's model becomes a job's reference model.

Data:

* gravity gz (positive down) and TMI (20 m up; 50 000 nT, I = 60, D = 10) on a 100 m grid,
  modelled on a finer mesh of their own (50 m x 50 m x 25 m), 2 % noise;
* DC dipole-dipole (a = 100 m, n = 1-4) on five east-west lines, 3 % noise;
* MT impedances (xy, yx) at 4 x 4 stations 500 m apart, 10, 100 and 1000 Hz, 3 % noise.

DC and MT are modelled on the inversion meshes themselves (an "inverse crime": the true model
fits them exactly); finer meshes would be far slower with SciPy's LU solver.

The jobs go through the data pipeline (worker.run_data_pipeline) as the upload page's jobs
do, on a coarse mesh: 100 m x 50 m cells to 800 m (MT: 200 m, 25 m at the top growing 10 % a
layer to 1 km, padding and air 6 km; finer, SciPy's LU solver takes many minutes a frequency).  The constrained runs get the geology spec of the holes alone
(builder.builder_spec of HOLES), the logs reaching 150 m around each hole.  For each run:
the misfit, the correlation of the recovered model with the true one over the core (to
600 m), and the mean value recovered in the two bodies; OUT/sections.png shows an east-west
section through both bodies (true, free, with the holes).

    python examples/synthetic_builder.py OUT                      # every method
    python examples/synthetic_builder.py OUT --only gravity dc    # some
    python examples/synthetic_builder.py OUT --quick              # 2 iterations each
    python examples/synthetic_builder.py OUT --only mt --ec2      # on one EC2 instance (PARDISO)
    python examples/synthetic_builder.py OUT --plot               # the figure of the runs in OUT

MT needs PARDISO: with SciPy's LU (a Mac) each model costs ~6 minutes of factorizations here,
and the MT runs take hours; ``--ec2`` runs the script on one instance in ap-south-1 (tagged
Owner = the IAM user; m5.2xlarge by default) and fetches its outputs into OUT, where the
methods run here already are: results.json is merged and the figure drawn with all of them.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoinv3d.datamodel.mesh import Mesh3D          # noqa: E402
from geoinv3d.datamodel.survey import SurveyData    # noqa: E402
from geoinv3d.methods.builder import builder_spec, model_values   # noqa: E402

AREA = [0.0, 2000.0, 0.0, 2000.0]
SECTION_Y = 1000.0
COVER, BASEMENT = 20.0, 1000.0          # ohm m
SIGMA_BG = 1 / 300.0                    # the forward simulations' (unused: they get the whole model)
INDUCING = [50000.0, 60.0, 10.0]
FREQUENCIES = [10.0, 100.0, 1000.0]
COMPONENTS = ["xy_real", "xy_imag", "yx_real", "yx_imag"]


def _p(density=None, susceptibility=None, resistivity=None):
    return {"density": {"value": density, "lower": None, "upper": None},
            "susceptibility": {"value": susceptibility, "lower": None, "upper": None},
            "resistivity": {"value": resistivity, "lower": None, "upper": None}}


def _unit(**kw):
    return {"weight": 10, "fixed": False, "sharp": False, **kw}


ORE = _unit(kind="body", name="sulphide", shape="box", box=[500.0, 750.0, 700.0, 1300.0],
            ref="depth", top=100.0, bottom=400.0, dip=60.0, dipdir=90.0,
            props=_p(3.3, 0.05, 3.0))
GRANITE = _unit(kind="body", name="granite", shape="cylinder", cyl={"x": 1450.0, "y": 1000.0, "r": 250.0},
                ref="depth", top=150.0, bottom=800.0, dip=90.0, dipdir=90.0,
                props=_p(2.57, None, 5000.0))
LAYERS = {"kind": "layers", "name": "Cover", "shape": "all", "ref": "depth", "top": 0.0, "dip": 0.0,
          "dipdir": 90.0, "origin": [1000.0, 1000.0],
          "layers": [_unit(name="cover", thickness=50.0, props=_p(resistivity=COVER), weight=5),
                     _unit(name="basement", thickness=None, props=_p(resistivity=BASEMENT), weight=5)]}
FREE = {"density": {"value": 2.67, "lower": None, "upper": None},
        "susceptibility": {"value": None, "lower": None, "upper": None},
        "resistivity": {"value": BASEMENT, "lower": None, "upper": None, "mixing": "log"}}


def _iv(a, b, name, rho, density=2.67, susceptibility=0.0):
    return {"from": a, "to": b, "lithology": name, "props": _p(density, susceptibility, rho)}


# the holes' logs, as the true model has it under them (BH1 leaves the dipping body at 360 m)
HOLES = {"kind": "boreholes", "name": "Holes", "radius": 150.0, "weight": 10, "fixed": False, "sharp": False,
         "holes": [{"name": "BH1", "x": 650.0, "y": 1000.0, "collar": None, "azimuth": 0.0, "dip": 90.0,
                    "intervals": [_iv(0, 50, "cover", COVER), _iv(50, 100, "basement", BASEMENT),
                                  _iv(100, 360, "sulphide", 3.0, 3.3, 0.05), _iv(360, 500, "basement", BASEMENT)]},
                   {"name": "BH2", "x": 1450.0, "y": 1000.0, "collar": None, "azimuth": 0.0, "dip": 90.0,
                    "intervals": [_iv(0, 50, "cover", COVER), _iv(50, 150, "basement", BASEMENT),
                                  _iv(150, 600, "granite", 5000.0, 2.57, 0.0)]}]}
MODEL = {"version": 2, "ground": 0.0, "items": [LAYERS, ORE, GRANITE], "free": FREE}
# what the constrained runs know: the holes; elsewhere the job's background (from the data)
KNOWN = {"version": 2, "ground": 0.0, "items": [HOLES],
         "free": {**FREE, "resistivity": {**FREE["resistivity"], "value": None}}}

# ── meshes, as the pipeline builds them ─────────────────────────────────

MESH = {"topography": {"flat_elevation": 0.0}, "mesh_type": "tensor",
        "core_cell_m": 100.0, "core_cell_z_m": 50.0, "depth_core_m": 800.0, "pad_distance_m": 1500.0}
MT_MESH = {"topography": {"flat_elevation": 0.0}, "mesh_type": "tensor",
           "core_cell_m": 200.0, "core_cell_z_m": 25.0, "depth_core_m": 1000.0,
           "pad_distance_m": 6000.0, "air_m": 6000.0, "margin_m": 250.0}


def _flat(x, y):
    return np.zeros(np.shape(x))


def tensor_mesh(extent, m=MESH):
    from geoinv3d.cloud.worker import _build_tensor_mesh
    return _build_tensor_mesh(extent, _flat, False, m["core_cell_m"], m["core_cell_z_m"],
                              m["depth_core_m"], m["pad_distance_m"])


def mt_mesh(extent):
    from geoinv3d.cloud.meshing import MT_TOP_LAYERS, MT_Z_GROWTH
    from geoinv3d.cloud.worker import _build_mt_tensor_mesh
    used = {k: MT_MESH[k] for k in ("core_cell_m", "core_cell_z_m", "depth_core_m", "pad_distance_m",
                                    "air_m", "margin_m")}
    used.update(mt=True, z_growth=MT_Z_GROWTH, top_layers=MT_TOP_LAYERS)
    return _build_mt_tensor_mesh(extent, _flat, False, used)


def cells(tm, active=None):
    a = np.ones(tm.n_cells, bool) if active is None else active
    return tm.cell_centers[a], tm.h_gridded[a] / 2.0


def truth(builder, prop, tm, active=None):
    c, h = cells(tm, active)
    return model_values(builder, prop, c, h)


def share(item, tm, active=None):
    """The share of each cell's volume an item of MODEL takes."""
    one = copy.deepcopy(item)
    one["props"] = _p(susceptibility=1.0)
    return truth({"version": 2, "ground": 0.0, "items": [one]}, "susceptibility", tm, active)


# ── data ────────────────────────────────────────────────────────────────

def _survey(locs, n):
    return SurveyData(locations=locs, observed=np.zeros(n), std=np.ones(n))


# DC surveys: dipole-dipole spacings (a, and the n of each) on five east-west lines, and the mesh
# each is inverted on (cells at most half the smallest a, so that the electrodes have cells between
# them).  "dc": a = 100 m, n = 1-4 — about 150 m deep, above most of the bodies; "dc50": a = 50 m,
# n = 1-8 — the cover resolved, still shallow; "dc_multi": a = 50 m and 200 m, n = 1-6 each — the
# short dipoles for the cover, the long ones (1.4 km apart at most) for the bodies.
DC_FINE = {**MESH, "core_cell_m": 25.0, "core_cell_z_m": 25.0, "depth_core_m": 600.0}
DC_SURVEYS = {"dc": ([(100.0, (1, 2, 3, 4))], MESH),
              "dc50": ([(50.0, tuple(range(1, 9)))], DC_FINE),
              "dc_multi": ([(50.0, tuple(range(1, 7))), (200.0, tuple(range(1, 7)))], DC_FINE)}


def dc_rows(key) -> np.ndarray:
    return np.vstack([dipole_dipole(a=a, ns=ns) for a, ns in DC_SURVEYS[key][0]])


def dc_extent(key):
    pts = dc_rows(key).reshape(-1, 3)
    return (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())


def dipole_dipole(lines=(200.0, 600.0, 1000.0, 1400.0, 1800.0), a=100.0, ns=(1, 2, 3, 4)):
    """(n, 12) electrode rows, consecutive rows sharing a source."""
    rows = []
    xs = np.arange(AREA[0], AREA[1] + 1e-9, a)
    for y in lines:
        for i in range(len(xs) - 1):
            for n in ns:
                j = i + 1 + n
                if j + 1 < len(xs):
                    rows.append([xs[i], y, 0, xs[i + 1], y, 0, xs[j], y, 0, xs[j + 1], y, 0])
    return np.array(rows)


def mt_stations():
    s = np.arange(250.0, 1751.0, 500.0)
    sx, sy = np.meshgrid(s, s)
    return np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])


def make_data(out: Path, only, seed: int = 0) -> dict:
    """The data files and the noise floors of the potential fields."""
    from geoinv3d.methods.dc_resistivity import DCResistivityMethod
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod
    from geoinv3d.methods.mt import MTMethod

    rng = np.random.default_rng(seed)
    info = {}
    if {"gravity", "magnetics"} & set(only):
        x = np.arange(-500.0, 2500.1, 50.0)
        pf = Mesh3D(hx=np.full(60, 50.0), hy=np.full(60, 50.0), hz=np.full(40, 25.0),
                    origin=(-500.0, -500.0, -1000.0))
        tm = pf.to_discretize()
        g = np.arange(AREA[0], AREA[1] + 1e-9, 100.0)
        gx, gy = np.meshgrid(g, g)
        for name, method, prop, z, header in (
                ("gravity", GravityMethod(), "density", 1.0, "gz"),
                ("magnetics", MagneticsMethod(inducing_field=tuple(INDUCING)), "susceptibility", 20.0, "tmi")):
            if name not in only:
                continue
            t0 = time.time()
            m = truth(MODEL, prop, tm)
            locs = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
            d = method.make_simulation_full(pf, _survey(locs, len(locs))).dpred(m)
            if name == "gravity":
                d = -d      # SimPEG's gz is z-up; field data are positive down
            floor = 0.02 * float(np.abs(d).max())
            d = d + rng.normal(scale=floor, size=d.size)
            np.savetxt(out / f"{name}.csv", np.column_stack([locs, d]), delimiter=",",
                       header=f"x,y,z,{header}", comments="", fmt="%.8g")
            info[name] = {"n_data": int(d.size), "peak": round(float(np.abs(d).max()), 4),
                          "noise_floor": floor, "seconds": round(time.time() - t0, 1)}
        del x
    for key in [k for k in DC_SURVEYS if k in only]:
        t0 = time.time()
        rows = dc_rows(key)
        mesh = tensor_mesh(dc_extent(key), DC_SURVEYS[key][1])
        m = truth(MODEL, "resistivity", mesh.to_discretize())
        d = DCResistivityMethod(sigma_background=SIGMA_BG).make_simulation_full(
            mesh, _survey(rows, len(rows))).dpred(m)
        std = 0.03 * np.abs(d) + 1e-3 * np.median(np.abs(d))
        d = d + std * rng.normal(size=d.size)
        np.savez(out / f"{key}.npz", electrodes=rows, values=d, std=std, data_type="volt")
        info[key] = {"n_data": int(d.size), "n_cells": mesh.n_cells, "seconds": round(time.time() - t0, 1),
                     "spacings": [{"a_m": a, "n": list(ns)} for a, ns in DC_SURVEYS[key][0]]}
    if "mt" in only:
        t0 = time.time()
        st = mt_stations()
        mesh = mt_mesh((st[:, 0].min(), st[:, 0].max(), st[:, 1].min(), st[:, 1].max()))
        tm = mesh.to_discretize()
        active = tm.cell_centers[:, 2] < 0
        m = truth(MODEL, "resistivity", tm, active)
        mt = MTMethod(frequencies=FREQUENCIES, components=COMPONENTS, sigma_background=SIGMA_BG,
                      primary_layers=([0.0, 50.0], [1 / COVER, 1 / BASEMENT]))
        n = len(FREQUENCIES) * len(COMPONENTS) * len(st)
        d = mt.make_simulation_active(mesh, _survey(st, n), active, forward_only=True).dpred(m)
        std = 0.03 * np.abs(d) + 0.01 * np.median(np.abs(d))
        d = d + std * rng.normal(size=d.size)
        np.savez(out / "mt.npz", locations=st, frequencies=FREQUENCIES, components=np.array(COMPONENTS),
                 values=d, std=std)
        info["mt"] = {"n_data": int(d.size), "n_cells": mesh.n_cells, "seconds": round(time.time() - t0, 1)}
    return info


# ── jobs ────────────────────────────────────────────────────────────────

ITER = {"param_mode": "manual", "max_iter": 15, "max_irls_iterations": 8, "beta0_ratio": 1.0,
        "cooling_factor": 2.0, "use_preconditioner": True}
METHODS = {
    "gravity": ("density", {"method_type": "gravity", "regularization_type": "sparse",
                            "bounds_lower": -0.5, "bounds_upper": 1.0}),
    "magnetics": ("susceptibility", {"method_type": "magnetic", "regularization_type": "sparse",
                                     "bounds_lower": 0.0, "bounds_upper": 0.2}),
    "dc": ("resistivity", {"method_type": "dc", "regularization_type": "l2", "max_iter": 8}),
    "dc50": ("resistivity", {"method_type": "dc", "regularization_type": "l2", "max_iter": 10}),
    "dc_multi": ("resistivity", {"method_type": "dc", "regularization_type": "l2", "max_iter": 10}),
    "mt": ("resistivity", {"method_type": "mt", "regularization_type": "l2", "max_iter": 10}),
}
QUICK = {"max_iter": 2, "max_irls_iterations": 1}


def job(key, info, constrained: bool, quick: bool) -> dict:
    prop, extra = METHODS[key]
    ds = {"gravity": {"method": "gravity", "files": ["gravity.csv"], "component": "gz", "noise_pct": 0.0},
          "magnetics": {"method": "magnetics", "files": ["magnetics.csv"], "component": "tmi", "noise_pct": 0.0,
                        "method_kwargs": {"inducing_field": INDUCING}},
          # (DC and MT: the background from the data's apparent resistivity, as the page's jobs)
          **{k: {"method": "dc", "files": [f"{k}.npz"]} for k in DC_SURVEYS},
          "mt": {"method": "mt", "files": ["mt.npz"]}}[key]
    if key in ("gravity", "magnetics"):
        ds["noise_floor"] = info[key]["noise_floor"]
    mesh = MT_MESH if key == "mt" else DC_SURVEYS[key][1] if key in DC_SURVEYS else MESH
    params = {"inversion_mode": "single", "datasets": [ds], **mesh,
              **ITER, **extra, **(QUICK if quick else {})}
    if constrained:
        params["geology"] = builder_spec(KNOWN, prop)
    return params


def compare(key, result, tm, active):
    """Misfit, correlation with the truth over the core and the bodies' recovered values."""
    prop = METHODS[key][0]
    rec = np.asarray(result["recovered_model"], float)
    true = truth(MODEL, prop, tm, active)
    c, _ = cells(tm, active)
    core = ((c[:, 0] >= AREA[0]) & (c[:, 0] <= AREA[1]) & (c[:, 1] >= AREA[2]) & (c[:, 1] <= AREA[3])
            & (c[:, 2] > -600.0))
    show = (lambda v: -v / np.log(10)) if prop == "resistivity" else (lambda v: v)   # log10 ohm m
    out = {"chi2_per_datum": None, "corr": round(float(np.corrcoef(show(rec[core]), show(true[core]))[0, 1]), 3)}
    d = result.get("data") or {}
    if d.get("predicted") is not None:
        out["chi2_per_datum"] = round(float(np.mean(((d["observed"] - d["predicted"]) / d["std"]) ** 2)), 2)
    for item in (ORE, GRANITE):
        s = share(item, tm, active) > 0.5
        if s.any():
            out[item["name"]] = {"true": round(float(np.mean(show(true[s]))), 3),
                                 "recovered": round(float(np.mean(show(rec[s]))), 3)}
    return out, rec, true


def run(out: Path, keys, info, quick: bool) -> dict:
    from geoinv3d.cloud.worker import run_data_pipeline
    results, sections = {}, {}
    for key in keys:
        for constrained in (False, True):
            label = f"{key} {'with the holes' if constrained else 'free'}"
            params = job(key, info, constrained, quick)
            t0 = time.time()
            # the pipeline's log as it goes (SimPEG's iterations too)
            with open(out / f"{key}_{'holes' if constrained else 'free'}.log", "w", buffering=1) as log, \
                    contextlib.redirect_stdout(log):
                result = run_data_pipeline(params, str(out))
            if key == "mt":
                st = mt_stations()
                tm = mt_mesh((st[:, 0].min(), st[:, 0].max(), st[:, 1].min(), st[:, 1].max())).to_discretize()
            elif key in DC_SURVEYS:
                tm = tensor_mesh(dc_extent(key), DC_SURVEYS[key][1]).to_discretize()
            else:
                tm = tensor_mesh(tuple(AREA)).to_discretize()
            if tm.n_cells != result["n_cells"]:
                raise RuntimeError(f"{label}: the pipeline's mesh has {result['n_cells']} cells, ours {tm.n_cells}")
            active = result.get("active_cells")
            active = None if active is None else np.asarray(active, bool)
            got, rec, true = compare(key, result, tm, active)
            got.update(seconds=round(time.time() - t0, 1), iterations=result.get("n_iterations"),
                       n_cells=int(result["n_active_cells"]), n_data=int(result["n_data"]))
            results[label] = got
            sections.setdefault(key, {"tm": tm, "active": active, "true": true})[constrained] = rec
            print(f"{label}: {json.dumps(got)}", flush=True)
        s = sections[key]
        np.savez(out / f"sections_{key}.npz", hx=s["tm"].h[0], hy=s["tm"].h[1], hz=s["tm"].h[2],
                 origin=s["tm"].origin, active=np.ones(s["tm"].n_cells, bool) if s["active"] is None else s["active"],
                 true=s["true"], free=s[False], holes=s[True])
    return results, sections


def load_sections(out: Path) -> dict:
    """The runs' models saved by run (sections_<method>.npz), in METHODS' order."""
    from discretize import TensorMesh
    sections = {}
    for key in METHODS:
        path = out / f"sections_{key}.npz"
        if path.exists():
            with np.load(path) as z:
                tm = TensorMesh([z["hx"], z["hy"], z["hz"]], origin=z["origin"])
                sections[key] = {"tm": tm, "active": z["active"], "true": z["true"],
                                 False: z["free"], True: z["holes"]}
    return sections


def save_results(out: Path, info: dict, runs: dict) -> None:
    """results.json, with what an earlier run (other methods, or another machine) left there."""
    path = out / "results.json"
    old = json.loads(path.read_text()) if path.exists() else {"data": {}, "runs": {}}
    old["data"].update(info)
    old["runs"].update(runs)
    path.write_text(json.dumps(old, indent=2))


def plot(out: Path, sections) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    rows = list(sections)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.9 * len(rows)), squeeze=False)
    names = {"density": "density contrast (g/cc)", "susceptibility": "susceptibility (SI)",
             "resistivity": "log10 resistivity (ohm m)"}
    for r, key in enumerate(rows):
        s, prop = sections[key], METHODS[key][0]
        tm, active = s["tm"], s["active"]
        j = int(np.searchsorted(tm.nodes_y, SECTION_Y) - 1)

        def slab(v):
            full = np.full(tm.n_cells, np.nan)
            full[np.ones(tm.n_cells, bool) if active is None else active] = v
            full = full.reshape(tm.shape_cells, order="F")[:, j, :].T
            return -full / np.log(10) if prop == "resistivity" else full

        panels = [slab(s["true"]), slab(s.get(False)), slab(s.get(True))]
        lo, hi = np.nanpercentile(panels[0], [1, 99]) if prop != "density" else (-0.65, 0.65)
        if prop == "resistivity":
            lo, hi = 0.0, 4.0
        cmap = {"density": "RdBu_r", "susceptibility": "viridis", "resistivity": "Spectral"}[prop]
        for c, (v, title) in enumerate(zip(panels, ("true", "free", "with the holes (BH1, BH2)"))):
            ax = axes[r, c]
            if v is None or np.all(np.isnan(v)):
                ax.set_visible(False)
                continue
            pm = ax.pcolormesh(tm.nodes_x, tm.nodes_z, v, cmap=cmap, vmin=lo, vmax=hi, shading="flat")
            ax.set_xlim(AREA[0] - 250, AREA[1] + 250)
            ax.set_ylim(-800, 0)
            ax.set_title(f"{key}: {title}", fontsize=9)
            # the bodies' true outlines on the section
            z0, z1 = -ORE["top"], -ORE["bottom"]
            dx = (ORE["bottom"] - ORE["top"]) / np.tan(np.radians(ORE["dip"]))
            ax.add_patch(Polygon([[500, z0], [750, z0], [750 + dx, z1], [500 + dx, z1]], fill=False, ec="k", lw=0.8))
            ax.plot([1200, 1200, 1700, 1700], [-150, -800, -800, -150], "k", lw=0.8)
            ax.plot([1200, 1700], [-150, -150], "k", lw=0.8)
            if c == 2:
                for h in HOLES["holes"]:
                    depth = max(iv["to"] for iv in h["intervals"])
                    ax.plot([h["x"], h["x"]], [0, -depth], color="m", lw=1.5)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("elevation (m)", fontsize=8)
        fig.colorbar(pm, ax=axes[r, :].tolist(), shrink=0.85, pad=0.01).set_label(names[prop], fontsize=8)
    fig.suptitle(f"East-west section at y = {SECTION_Y:g} m: a model from the model builder, "
                 "its synthetic data inverted on a coarse mesh", fontsize=10)
    path = out / "sections.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def on_ec2(args) -> int:
    """This script on one EC2 instance (PARDISO: MT in minutes rather than hours), its outputs
    fetched into OUT (results.json merged), then the instance terminated (after checking it is
    ours), also after errors."""
    sys.path.insert(0, str(REPO / "deploy"))
    import ec2_sweep as es
    from geoinv3d.cloud.ec2 import REMOTE, EC2Backend, code_archive
    b = EC2Backend(args.region)
    if args.attach:          # a run started before: follow it, fetch, terminate
        iid = args.attach
        print(time.strftime("%H:%M:%S"), "following", iid, flush=True)
    else:
        iid = es.launch(b, f"synthbuilder-{int(time.time())}", 0, args.type)
        print(time.strftime("%H:%M:%S"), "launched", iid, args.type, args.region, "owner", b.owner, flush=True)
    try:
        ssh = es.ssh_to(b, iid)
        while not args.attach and ssh.read(f"{REMOTE}/READY") is None:
            if ssh.read(f"{REMOTE}/BOOTSTRAP_FAILED") is not None:
                print(ssh.run("sudo tail -n 30 /var/log/geoinv3d-bootstrap.log")[1])
                return 2
            time.sleep(10)
        if not args.attach:
            print(time.strftime("%H:%M:%S"), "set up", flush=True)
            # the boot-time shutdown waits for a job; this is none: our own limit instead
            ssh.run(f"sudo shutdown -c; sudo shutdown -h +{args.timeout_min}")
            ssh.write(f"{REMOTE}/code.tar.gz", code_archive())
            ssh.run(f"rm -rf {REMOTE}/code/geoinv3d && tar xzf {REMOTE}/code.tar.gz -C {REMOTE}/code")
            ssh.write(f"{REMOTE}/synthetic_builder.py", Path(__file__).read_bytes())
            flags = f"--only {' '.join(args.only)}" + (" --quick" if args.quick else "")
            inner = (f"cd {REMOTE} && PYTHONPATH={REMOTE}/code {REMOTE}/venv/bin/python -W ignore -u "
                     f"synthetic_builder.py out {flags} > run.log 2>&1; echo $? > run.done")
            ssh.run(f"{{ nohup setsid bash -c '{inner}' > /dev/null 2>&1 < /dev/null & }} && echo started")
        shown, last = 0, ""
        while True:
            time.sleep(20)
            lines = [ln for ln in (ssh.read(f"{REMOTE}/run.log") or b"").decode(errors="replace").splitlines()
                     if ln.strip() and "INFO" not in ln]
            for ln in lines[shown:]:
                print(ln, flush=True)
            shown = len(lines)
            # the iteration the pipeline is at (its log as it goes; no log yet: nothing — a bare
            # `tail` would wait on the channel's input for ever)
            _, tail = ssh.run(f"f=$(ls -t {REMOTE}/out/*.log 2>/dev/null | head -1); "
                              f"[ -n \"$f\" ] && tail -n 1 \"$f\" < /dev/null; true")
            tail = tail.strip()
            if tail and tail != last and tail[:4].strip().isdigit():
                print("   ", tail[:110], flush=True)
                last = tail
            done = ssh.read(f"{REMOTE}/run.done")
            if done is not None:
                break
        args.out.mkdir(parents=True, exist_ok=True)
        _, names = ssh.run(f"cd {REMOTE}/out && ls")
        for name in names.split():
            if name.endswith((".npz", ".log", ".json")) and name not in ("model.json",) and not name.startswith(("dc.", "mt.")):
                data = ssh.read(f"{REMOTE}/out/{name}")
                if name == "results.json":
                    got = json.loads(data)
                    save_results(args.out, got["data"], got["runs"])
                else:
                    (args.out / name).write_bytes(data)
        print(time.strftime("%H:%M:%S"), "fetched into", args.out, "exit", done.decode().strip(), flush=True)
        return int(done.decode().strip() or 1)
    finally:
        inst = b._instance(iid) or {}
        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
        if tags.get("Owner") == b.owner and tags.get("Project") == "geoinv3d":
            b.ec2.terminate_instances(InstanceIds=[iid])
            print(time.strftime("%H:%M:%S"), "terminated", iid, f"(Owner={tags['Owner']})", flush=True)
        else:
            print(f"NOT terminated: {iid} has tags {tags}; check it in the console", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--only", nargs="*", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--quick", action="store_true", help="2 iterations each: every path, quickly")
    p.add_argument("--ec2", action="store_true", help="run on one EC2 instance (MT with PARDISO), then terminate it")
    p.add_argument("--plot", action="store_true", help="only draw OUT/sections.png from the runs saved in OUT")
    p.add_argument("--type", default="m5.2xlarge", help="with --ec2: the instance type")
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--timeout-min", type=int, default=90, help="with --ec2: it shuts down after this")
    p.add_argument("--attach", default=None, help="with --ec2: follow the run on this instance (started "
                   "before), fetch its outputs and terminate it")
    args = p.parse_args()
    if args.ec2:
        code = on_ec2(args)
        print("figure:", plot(args.out, load_sections(args.out)))
        return code
    if args.plot:
        print("figure:", plot(args.out, load_sections(args.out)))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    # the model as the page saves it (with the holes), to open in /model
    (args.out / "model.json").write_text(json.dumps(
        {"name": "Synthetic: sulphide body and granite", "crs": "", "extent": AREA, "ground": 0.0,
         "depth": 1000.0, "ve": 1.0, "mesh": {"dx": MESH["core_cell_m"], "dz": MESH["core_cell_z_m"]},
         "builder": {**MODEL, "items": MODEL["items"] + [HOLES]}}, indent=1))
    t0 = time.time()
    info = make_data(args.out, args.only)
    print("data:", json.dumps(info), flush=True)
    results, _ = run(args.out, args.only, info, args.quick)
    save_results(args.out, info, results)
    try:
        print("figure:", plot(args.out, load_sections(args.out)))
    except ImportError:          # (no matplotlib on the EC2 image: drawn where the outputs go)
        print("no matplotlib here: draw the figure with --plot")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
