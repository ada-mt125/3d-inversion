"""End-to-end check of the EC2 backend on real AWS with a synthetic gravity survey.

    python deploy/ec2_smoke_test.py                       # tiny: 7 x 7 stations, ~1000 cells
    python deploy/ec2_smoke_test.py --width 1000 --spacing 25 --auto-mesh \\
        --regularization sparse --instance c5.2xlarge     # denser survey and mesh

Launches ONE instance (tagged Owner=<your IAM user>, Project=geoinv3d),
installs the software, runs the inversion, downloads the result and
terminates the instance.  The instance is terminated in every case,
including errors and Ctrl+C.  Creates (once, reused later) the key pair
geoinv3d-<user> and the security group geoinv3d-ssh-<user> (SSH from this
machine's IP only); both are free.

The synthetic model is a 0.3 g/cc block centred in the survey, from 17% to
50% of the survey width deep (for 600 m: 200-400 m wide, 100-300 m deep).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoinv3d.cloud.ec2 import EC2Backend, TERMINAL_PHASES   # noqa: E402

DENSITY = 0.3


def block_mask(cc, width):
    c, half = width / 2, width / 6
    return ((abs(cc[:, 0] - c) < half) & (abs(cc[:, 1] - c) < half)
            & (cc[:, 2] < -width / 6) & (cc[:, 2] > -width / 2))


def synthetic_gravity_csv(path: Path, width: float, spacing: float, noise: float) -> int:
    """Stations every ``spacing`` m over a ``width`` m square above the block."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    axis = np.arange(0.0, width + 1e-6, spacing)
    xx, yy = np.meshgrid(axis, axis)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    h = width / 30   # forward-model cells: the block is 10 x 10 x 10 of them
    n = 50
    mesh = Mesh3D.uniform(n, n, 30, h, h, h, origin=(width / 2 - n * h / 2,) * 2 + (-30 * h,))
    model = block_mask(mesh.to_discretize().cell_centers, width) * DENSITY
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    # field convention (positive downward), like real Bouguer data
    gz = -GravityMethod().make_simulation(mesh, survey).dpred(model)
    gz = gz + np.random.default_rng(0).normal(scale=noise * np.abs(gz).max(), size=gz.size)
    rows = "\n".join(f"{x:.2f},{y:.2f},{v:.6f}" for (x, y, _), v in zip(locs, gz))
    path.write_text("x,y,gz\n" + rows + "\n", encoding="utf-8")
    return len(locs)


def recovery(result_dir: Path, width: float) -> dict:
    """Peak, depth and share of the recovered density inside the true block."""
    import discretize
    meta = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(result_dir / "result.zip") as zf:
        m = np.load(io.BytesIO(zf.read("recovered_model.npy")))
        active = np.load(io.BytesIO(zf.read("active_cells.npy"))) \
            if "active_cells.npy" in zf.namelist() else None
    mesh = getattr(discretize, meta["mesh"]["__class__"]).deserialize(meta["mesh"])
    cc, vol = mesh.cell_centers, mesh.cell_volumes
    if active is not None:
        cc, vol = cc[active], vol[active]
    pos = np.clip(m, 0, None) * vol
    inside = block_mask(cc, width)
    return {"peak": float(m.max()), "peak_depth_m": float(-cc[np.argmax(m), 2]),
            "centroid_depth_m": float(-(pos @ cc[:, 2]) / pos.sum()),
            "share_in_block": float(pos[inside].sum() / pos.sum()),
            "n_cells": meta.get("n_cells"), "mesh_design": meta.get("mesh_design", {}).get("used")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--instance", default="c5.xlarge")
    ap.add_argument("--width", type=float, default=600.0, help="survey width (m)")
    ap.add_argument("--spacing", type=float, default=100.0, help="station spacing (m)")
    ap.add_argument("--noise", type=float, default=0.01, help="noise, fraction of max |gz|")
    ap.add_argument("--auto-mesh", action="store_true",
                    help="let the pipeline design the mesh from the data spacing")
    ap.add_argument("--regularization", default="l2", choices=["l2", "sparse", "l1l2"])
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--timeout-min", type=float, default=40)
    args = ap.parse_args()

    backend = EC2Backend(args.region)
    print(f"Owner tag: {backend.owner}, region {args.region}, instance {args.instance}")
    task_id = f"smoke{int(time.time())}"
    params = {"method_type": "gravity", "inversion_mode": "single",
              "datasets": [{"method": "gravity", "files": ["g.csv"], "noise_pct": 0.0,
                            "noise_floor": 0.0}],
              "param_mode": "manual", "regularization_type": args.regularization,
              "max_iter": args.max_iter, "max_irls_iterations": 20,
              "topography": {"flat_elevation": 0}, "mesh_type": "tensor", "task_id": task_id}
    if not args.auto_mesh:
        params.update(core_cell_m=100.0, core_cell_z_m=100.0, depth_core_m=400.0,
                      pad_distance_m=200.0)
    job_id = None
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "g.csv"
            n = synthetic_gravity_csv(data, args.width, args.spacing, args.noise)
            gz = np.loadtxt(data, delimiter=",", skiprows=1)[:, 2]
            params["datasets"][0]["noise_floor"] = max(args.noise * np.abs(gz).max(), 1e-4)
            print(f"{n} stations every {args.spacing:g} m over {args.width:g} m")
            job_id = backend.start_job(task_id, [str(data)], params, args.instance)
        print(f"Launched {job_id} ({task_id})")
        record = {"job_id": job_id, "task_id": task_id, **backend.initial_fields()}
        last = None
        while time.time() - t0 < args.timeout_min * 60:
            record.update(backend.refresh(record))
            p = record.get("progress") or {}
            line = (f"{record['phase']:13s} {p.get('stage', '')} "
                    f"{('it ' + str(p['iteration'])) if p.get('iteration') is not None else ''}")
            if line != last:
                print(f"  {time.time() - t0:6.0f} s  {line}  {record.get('refresh_error') or ''}",
                      flush=True)
                last = line
            if record["phase"] in TERMINAL_PHASES or record["phase"] == "stopped":
                break
            time.sleep(15)
        summary = backend.result_summary(record)
        print(f"Finished: {record['phase']} after {time.time() - t0:.0f} s"
              + (f" ({record.get('reason')})" if record.get("reason") else ""))
        if summary:
            its = summary.get("iterations") or [{}]
            print(f"  {summary.get('n_iterations')} iterations, final phi_d "
                  f"{its[-1].get('phi_d')}, {summary.get('n_data')} data, "
                  f"error: {summary.get('error')}")
            print(f"  result files in {backend.result_dir(record)}")
            if record["phase"] == "succeeded":
                rec = recovery(backend.result_dir(record), args.width)
                print(f"  recovery: {json.dumps(rec)}")
        return 0 if record["phase"] == "succeeded" else 1
    finally:
        if job_id is not None:
            backend.ec2.terminate_instances(InstanceIds=[job_id])
            print(f"Terminated {job_id} (safety net; harmless if already terminated)")


if __name__ == "__main__":
    sys.exit(main())
