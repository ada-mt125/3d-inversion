"""Synthetic gravity, magnetic, DC resistivity and MT data of one body, and jobs for them.

One block 200 m x 200 m, 75-225 m deep, is dense (+0.3 g/cc), magnetic
(0.03 SI) and conductive (0.1 S/m in 0.01 S/m).  Its data:

* gravity gz on a 13 x 13 grid (50 m), field convention (positive down);
* TMI on the same grid, 20 m above ground (50 000 nT, I = 60, D = 10);
* DC dipole-dipole (a = 50 m, n = 1-4) on seven lines, potential differences;
* MT impedances (xy, yx; real and imaginary) at 3 x 3 stations (300 m apart), 100 and 1000 Hz.

All cover the same 600 m x 600 m, so every job gets the same inversion mesh
(``MESH``).  Gravity and TMI are modelled on a finer mesh of their own; DC and
MT on the inversion mesh itself (the true model fits them exactly, an
"inverse crime"): their discretization error on 50 m cells is far above the
noise (DC with 50 m dipoles: chi^2 of the true model ~ 50 N), and cells small
enough for it would make the MT jobs slow.  Noise: 2-3 % Gaussian.  MT is
kept small (72 data): with SimPEG's LU solver one forward run on this mesh
takes ~30 s and a Jacobian ~40 s, so the MT jobs have fewer iterations.  ``--local`` runs the jobs here with the data pipeline;
``--submit`` sends them to the local API server (GeoInv3D.command) in a new
workspace (or ``--workspace-id``), whose finished runs then show as one
workflow on the page.  Jobs: one single inversion per method, and the joint
gravity + magnetics inversion by both joint methods.

    python examples/synthetic_multiphysics.py OUT_DIR                 # data + jobs.json
    python examples/synthetic_multiphysics.py OUT_DIR --local [NAMES] # run here
    python examples/synthetic_multiphysics.py OUT_DIR --submit [NAMES] [--workspace-id ID]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoinv3d.datamodel.mesh import Mesh3D          # noqa: E402
from geoinv3d.datamodel.survey import SurveyData    # noqa: E402

INDUCING = [50000.0, 60.0, 10.0]
SIGMA_BG, SIGMA_BODY = 1e-2, 1e-1
FREQUENCIES = [100.0, 1000.0]
COMPONENTS = ["xy_real", "xy_imag", "yx_real", "yx_imag"]
BODY = dict(x=(-100.0, 100.0), y=(-100.0, 100.0), z=(-225.0, -75.0))


def _padded_mesh(h, n_core_xy, n_core_z, n_pad):
    from discretize import TensorMesh
    xy = [(h, n_pad, -1.4), (h, n_core_xy), (h, n_pad, 1.4)]
    tm = TensorMesh([xy, xy, [(h, n_pad, -1.4), (h, n_core_z)]], origin="CCN")
    return tm, Mesh3D(hx=tm.h[0], hy=tm.h[1], hz=tm.h[2], origin=tuple(tm.origin))


def _body(cc):
    return ((cc[:, 0] > BODY["x"][0]) & (cc[:, 0] < BODY["x"][1])
            & (cc[:, 1] > BODY["y"][0]) & (cc[:, 1] < BODY["y"][1])
            & (cc[:, 2] > BODY["z"][0]) & (cc[:, 2] < BODY["z"][1]))


def _survey(locs, n):
    return SurveyData(locations=locs, observed=np.zeros(n), std=np.ones(n))


EXTENT = (-300.0, 300.0, -300.0, 300.0)


def inversion_mesh():
    """The mesh the pipeline builds for these data and MESH (flat ground at z = 0)."""
    from geoinv3d.cloud.worker import _build_tensor_mesh
    flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))   # noqa: E731
    return _build_tensor_mesh(EXTENT, flat, False, MESH["core_cell_m"], MESH["core_cell_z_m"],
                              MESH["depth_core_m"], MESH["pad_distance_m"])


def dipole_dipole(lines=(-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0), a=50.0,
                  half_length=300.0,
                  ns=(1, 2, 3, 4)) -> np.ndarray:
    """(n, 12) electrode rows, consecutive rows sharing a source."""
    rows = []
    for y in lines:
        xs = np.arange(-half_length, half_length + 1e-9, a)
        for i in range(len(xs) - 1):
            for n in ns:
                j = i + 1 + n
                if j + 1 < len(xs):
                    rows.append([xs[i], y, 0, xs[i + 1], y, 0, xs[j], y, 0, xs[j + 1], y, 0])
    return np.array(rows)


def make_data(out: Path, seed: int = 0) -> dict:
    """Write the four data files and the true model; returns their summaries."""
    from geoinv3d.methods.dc_resistivity import DCResistivityMethod
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod
    from geoinv3d.methods.mt import MTMethod

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary = {}

    # potential fields: 25 m cells, no padding needed
    pf = Mesh3D.uniform(40, 40, 16, 25.0, 25.0, 25.0, origin=(-500.0, -500.0, -400.0))
    body = _body(pf.to_discretize().cell_centers)
    xy = np.arange(-300.0, 300.1, 50.0)
    xx, yy = np.meshgrid(xy, xy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for name, method, m, z, rel, header in (
            ("gravity", GravityMethod(), 0.3 * body, 1.0, 0.02, "gz"),
            ("magnetics", MagneticsMethod(inducing_field=tuple(INDUCING)), 0.03 * body, 20.0,
             0.02, "tmi")):
        locs = np.column_stack([grid, np.full(len(grid), z)])
        d = method.make_simulation_full(pf, _survey(locs, len(locs))).dpred(m)
        if name == "gravity":
            d = -d     # SimPEG's gz is z-up; field data are positive down
        d = d + rng.normal(scale=rel * abs(d).max(), size=d.size)
        rows = np.column_stack([locs, d])
        np.savetxt(out / f"{name}.csv", rows, delimiter=",", header=f"x,y,z,{header}",
                   comments="", fmt="%.8g")
        summary[name] = {"n_data": int(d.size), "peak": float(abs(d).max())}

    # DC and MT: log-conductivity on the inversion mesh
    mesh = inversion_mesh()
    tm = mesh.to_discretize()
    m_true = np.where(_body(tm.cell_centers), np.log(SIGMA_BODY), np.log(SIGMA_BG))
    np.savez(out / "true_log_conductivity.npz", values=m_true)
    rows = dipole_dipole()
    dc = DCResistivityMethod(sigma_background=SIGMA_BG)
    d = dc.make_simulation_full(mesh, _survey(rows, len(rows))).dpred(m_true)
    std = 0.03 * abs(d) + 1e-3 * np.median(abs(d))
    d = d + std * rng.normal(size=d.size)
    np.savez(out / "dc.npz", electrodes=rows, values=d, std=std, data_type="volt")
    summary["dc_resistivity"] = {"n_data": int(d.size),
                                 "n_sources": int(len(np.unique(rows[:, :6], axis=0)))}

    s = np.array([-300.0, 0.0, 300.0])
    sx, sy = np.meshgrid(s, s)
    stations = np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])
    mt = MTMethod(frequencies=FREQUENCIES, components=COMPONENTS, sigma_background=SIGMA_BG)
    n = len(FREQUENCIES) * len(COMPONENTS) * len(stations)
    d = mt.make_simulation_full(mesh, _survey(stations, n)).dpred(m_true)
    std = 0.03 * abs(d) + 0.01 * np.median(abs(d))
    d = d + std * rng.normal(size=d.size)
    np.savez(out / "mt.npz", locations=stations, frequencies=FREQUENCIES,
             components=np.array(COMPONENTS), values=d, std=std)
    summary["mt"] = {"n_data": int(d.size), "n_stations": int(len(stations))}
    return summary


# ── jobs ────────────────────────────────────────────────────────────────

# The errors of the potential-field data: the 2 % of their peaks that make_data adds
# (0.41 mGal and 164 nT).  (The first AWS run used 0.02 mGal and 1.5 nT: gravity was
# then fitted below its noise and TMI above it, whatever the inversion.)
DATASETS = {
    "gravity": {"method": "gravity", "files": ["gravity.csv"], "component": "gz",
                "noise_pct": 0.0, "noise_floor": 0.0081},
    "magnetics": {"method": "magnetics", "files": ["magnetics.csv"], "component": "tmi",
                  "noise_pct": 0.0, "noise_floor": 3.3,
                  "method_kwargs": {"inducing_field": INDUCING}},
    "dc": {"method": "dc", "files": ["dc.npz"],
           "method_kwargs": {"sigma_background": SIGMA_BG}},
    "mt": {"method": "mt", "files": ["mt.npz"],
           "method_kwargs": {"sigma_background": SIGMA_BG}},
}
MESH = {"topography": {"flat_elevation": 0.0}, "mesh_type": "tensor",
        "core_cell_m": 50.0, "core_cell_z_m": 50.0, "depth_core_m": 400.0,
        "pad_distance_m": 1500.0}
ITER = {"param_mode": "manual", "max_iter": 20, "max_irls_iterations": 10,
        "beta0_ratio": 1.0, "cooling_factor": 2.0, "use_preconditioner": True}


def jobs() -> list[dict]:
    """The runs: one per method, then the joint inversions (both methods)."""
    def ds(key, **extra):
        return {**DATASETS[key], **extra}

    single = [
        ("gravity", "gravity", "sparse", {"bounds_lower": -0.5, "bounds_upper": 1.0}),
        ("magnetics", "magnetic", "sparse", {"bounds_lower": 0.0, "bounds_upper": 0.2}),
        ("dc", "dc", "l2", {}),
        ("mt", "mt", "l2", {"max_iter": 10, "max_irls_iterations": 5}),
    ]
    out = []
    for key, method, reg, extra in single:
        out.append({"name": f"{key} ({reg})", "files": DATASETS[key]["files"],
                    "instance_type": "c5.xlarge",
                    "params": {"method_type": method, "inversion_mode": "single",
                               "datasets": [ds(key)], "regularization_type": reg,
                               **MESH, **ITER, **extra}})
    # Joint gravity + magnetics (the joint inversions with MT and DC were set aside:
    # with SimPEG's single-threaded LU solver their MT runs take hours on EC2)
    two = ["gravity.csv", "magnetics.csv"]
    pf_sets = [ds("gravity"), ds("magnetics")]
    # method one: sparse density and susceptibility, each with its bounds
    per_model = [{"regularization_type": "sparse", "bounds_lower": -0.5, "bounds_upper": 1.0},
                 {"regularization_type": "sparse", "bounds_lower": 0.0, "bounds_upper": 0.2}]
    out.append({"name": "joint gravity + magnetics: per-model regularization", "files": two,
                "instance_type": "c5.xlarge",
                "params": {"method_type": "joint", "inversion_mode": "joint",
                           "datasets": [dict(d, regularization=r)
                                        for d, r in zip(pf_sets, per_model)],
                           "regularization_type": "sparse", "cross_gradient_weight": 0.0,
                           **MESH, **ITER}})
    # method two: L2 + group lasso (Utsugi 2025) with a cross-gradient
    out.append({"name": "joint gravity + magnetics: group lasso + cross-gradient",
                "files": two, "instance_type": "c5.xlarge",
                "params": {"method_type": "joint", "inversion_mode": "joint",
                           "datasets": pf_sets, "regularization_type": "group_lasso",
                           "gl_n_lambda1": 8, "gl_lambda1_decades": 2.5, "gl_lambda2": 0.3,
                           "gl_cross_gradient": 0.1, **MESH, "param_mode": "manual"}})
    return out


QUICK = {"max_iter": 2, "max_irls_iterations": 1, "gl_n_lambda1": 1, "gl_gn_max_iter": 1}


def run_local(out: Path, names=None, quick: bool = False) -> dict:
    """Run the jobs here; ``quick`` cuts the iterations to check every path."""
    from geoinv3d.cloud.worker import pack_result, run_data_pipeline
    results = {}
    for job in jobs():
        if names and not any(n in job["name"] for n in names):
            continue
        params = {**job["params"], **(QUICK if quick else {})}
        t0 = time.time()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            result = run_data_pipeline(params, str(out))
        slug = job["name"].split(":")[0].split(" ")[0] + ("_" + job["params"].get(
            "regularization_type", "") if job["params"]["inversion_mode"] == "joint" else "")
        pack_result(result, str(out / f"{slug}_result.zip"))
        chi2 = result.get("chi2") or (result.get("group_lasso") or {}).get("chi2")
        if chi2 is None and result.get("data"):
            d = result["data"]
            chi2 = float(np.sum(((d["observed"] - d["predicted"]) / d["std"]) ** 2))
        results[job["name"]] = {"seconds": round(time.time() - t0, 1),
                                "n_cells": result["n_cells"], "chi2": chi2,
                                "n_data": result["n_data"], "notes": result.get("notes")}
        print(f"{job['name']}: {results[job['name']]}", flush=True)
    return results


def submit(out: Path, url: str = "http://127.0.0.1:8000",
           workspace: str = "Synthetic multiphysics", names=None,
           workspace_id: str | None = None) -> dict:
    """Send the jobs (or those whose name contains one of ``names``) to the local API
    server, in a new workspace or the one with ``workspace_id``."""
    import urllib.request
    import uuid

    def call(method, path, body=None, headers=None):
        req = urllib.request.Request(url + path, data=body, method=method,
                                     headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    ws_id = workspace_id
    if not ws_id:
        ws = call("POST", "/api/workspaces", json.dumps({"name": workspace}).encode(),
                  {"Content-Type": "application/json"})
        ws_id = ws["id"]
    submitted = []
    for job in jobs():
        if names and not any(n in job["name"] for n in names):
            continue
        boundary = uuid.uuid4().hex
        parts = []
        fields = {"method": job["params"]["method_type"], "mesh_type": "tensor",
                  "instance_type": job["instance_type"], "note": job["name"],
                  "params_json": json.dumps(job["params"]), "workspace_id": ws_id}
        for k, v in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                         f"{v}\r\n".encode())
        for name in job["files"]:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                         f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
                         .encode() + (out / name).read_bytes() + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        r = call("POST", "/api/inversion/submit", body,
                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        submitted.append({"name": job["name"], "job_id": r["job_id"]})
        print(f"submitted {job['name']}: {r['job_id']}", flush=True)
    return {"workspace_id": ws_id, "jobs": submitted}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--local", nargs="*", default=None, help="run jobs here (optionally by name)")
    p.add_argument("--quick", action="store_true", help="with --local: 1-2 iterations each")
    p.add_argument("--regenerate", action="store_true", help="make the data again")
    p.add_argument("--submit", nargs="*", default=None,
                   help="send the jobs (optionally by name) to the local API server")
    p.add_argument("--workspace-id", default=None, help="with --submit: an existing workspace")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    args = p.parse_args()
    if args.regenerate or not (args.out / "mt.npz").exists():
        print(json.dumps(make_data(args.out), indent=2))
    (args.out / "jobs.json").write_text(json.dumps(jobs(), indent=2))
    if args.local is not None:
        run_local(args.out, args.local, args.quick)
    if args.submit is not None:
        print(json.dumps(submit(args.out, args.url, names=args.submit,
                                workspace_id=args.workspace_id), indent=2))


if __name__ == "__main__":
    main()
