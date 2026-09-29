"""Run one dataset with several regularizations in parallel on EC2, then compare.

    python deploy/ec2_multi_run.py karnataka-gravity [--only sparse,l1l2]
    python deploy/ec2_multi_run.py karnataka-gravity --resume karnataka-gravity-1790537566

``--resume STAMP`` follows the running instances of an earlier call (found by
their geoinv3d:task tag) instead of launching new ones: a job that already
started is followed to the end, one that did not is set up again.

Each variant runs on its own instance (EC2 backend, tagged Owner=<user>);
progress is printed line by line, the results are downloaded to
~/.geoinv3d/results/<task>/, and all variants go into one DAG viewer page
(examples/output/ec2_runs/<name>.geoinv3d_viewer.html).  Every launched
instance is terminated at the end, also after errors or Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoinv3d.cloud.ec2 import EC2Backend, TERMINAL_PHASES   # noqa: E402

PRICE = {"c5.xlarge": 0.17, "c5.2xlarge": 0.34, "c5.4xlarge": 0.68, "c5.9xlarge": 1.53}
DESKTOP = Path.home() / "OneDrive - Imperial College London" / "Desktop"

# Karnataka-AP gravity (NGPM Bouguer grid, EPSG:32643), the AOI of the earlier
# inversions, decimated to 1 km, second-order trend surface removed.
KARNATAKA_GRAVITY = {
    "files": [DESKTOP / "karnataka_ap_gravity" / "GEOTIFF" / "NGPM_BA.tiff"],
    "base": {
        "method_type": "gravity", "inversion_mode": "single",
        "datasets": [{"method": "gravity", "files": ["NGPM_BA.tiff"], "component": "gz",
                      "noise_pct": 0.0, "noise_floor": 0.5,
                      "regional": {"method": "polynomial", "order": 2}}],
        "aoi": [641000, 711000, 1634000, 1704000], "decimate_stride": 2,
        "topography": {"flat_elevation": 0}, "mesh_type": "tensor",
        "core_cell_m": 1000.0, "core_cell_z_m": 500.0, "depth_core_m": 10000.0,
        "pad_distance_m": 20000.0,
        "param_mode": "manual", "max_iter": 30, "max_irls_iterations": 30,
        "beta0_ratio": 1.0, "cooling_factor": 2.0, "use_preconditioner": True,
        # density contrast range of the rock samples (2.52-3.40 g/cc, median 2.66)
        "bounds_lower": -0.2, "bounds_upper": 0.5,
    },
    "variants": {
        "sparse": ({"regularization_type": "sparse", "norms": [0, 2, 2, 1],
                    "alpha_s": 1e-4, "alpha_x": 1, "alpha_y": 1, "alpha_z": 1},
                   "c5.2xlarge"),
        # Settings recommended by the depth study (compact, Li & Oldenburg depth
        # weighting): alpha_s = 1 is the most compact; 0.1 keeps off the bounds
        "sparse_as1_dw1": ({"regularization_type": "sparse", "norms": [0, 2, 2, 2],
                            "alpha_s": 1.0, "alpha_x": 1, "alpha_y": 1, "alpha_z": 1,
                            "depth_weighting": "depth", "depth_weighting_exponent": 1.0},
                           "c5.2xlarge"),
        "sparse_as01_dw1": ({"regularization_type": "sparse", "norms": [0, 2, 2, 2],
                             "alpha_s": 0.1, "alpha_x": 1, "alpha_y": 1, "alpha_z": 1,
                             "depth_weighting": "depth", "depth_weighting_exponent": 1.0},
                            "c5.2xlarge"),
        # IRLS: 3.5 min on c5.4xlarge for 5040 data x 190k cells
        "l1l2": ({"regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "irls",
                  "l1l2_weighting": "S1", "beta_selection": "auto"}, "c5.4xlarge"),
        # Utsugi's CDA path: single-threaded, and each lambda point took about twice
        # as long as the one before (11 of 41 points in 11 min), so impractical here
        "l1l2_cda": ({"regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "cda",
                      "l1l2_weighting": "S1", "lambda_decades": 4, "beta_selection": "auto"},
                     "c5.4xlarge"),
        "l2": ({"regularization_type": "l2", "alpha_s": 1e-4, "alpha_x": 1, "alpha_y": 1,
                "alpha_z": 1}, "c5.2xlarge"),
        "mgs": ({"regularization_type": "mgs", "alpha_s": 1e-4, "alpha_x": 1, "alpha_y": 1,
                 "alpha_z": 1, "focusing_percentile": 95}, "c5.2xlarge"),
        "tv": ({"regularization_type": "tv", "alpha_s": 1e-4, "alpha_x": 1, "alpha_y": 1,
                "alpha_z": 1, "focusing_percentile": 95}, "c5.2xlarge"),
    },
}

# Synthetic magnetic test (examples/synthetic_magnetic_karnataka.py): two blocks and a dipping
# dyke under the Karnataka IGRF 2020 field, on the mesh of the gravity runs; susceptibility >= 0
_MAG_FIELD = [42100.0, 19.3, -1.4]
_MAG_SPARSE = {"regularization_type": "sparse", "norms": [0, 2, 2, 2], "alpha_s": 1.0,
               "alpha_x": 1, "alpha_y": 1, "alpha_z": 1}
SYNTHETIC_MAGNETIC = {
    "files": [REPO / "examples" / "output" / "synthetic_magnetic" / "karnataka_mag_synthetic.csv"],
    "base": {
        "method_type": "magnetics", "inversion_mode": "single",
        "datasets": [{"method": "magnetics", "files": ["karnataka_mag_synthetic.csv"],
                      "component": "tmi", "method_kwargs": {"inducing_field": _MAG_FIELD},
                      "noise_pct": 0.02, "noise_floor": 1.0}],
        "method_kwargs": {"inducing_field": _MAG_FIELD},
        "topography": {"flat_elevation": 0}, "mesh_type": "tensor",
        "core_cell_m": 1000.0, "core_cell_z_m": 500.0, "depth_core_m": 10000.0,
        "pad_distance_m": 20000.0,
        "param_mode": "manual", "max_iter": 30, "max_irls_iterations": 30,
        "beta0_ratio": 1.0, "cooling_factor": 2.0, "use_preconditioner": True,
        "bounds_lower": 0.0, "bounds_upper": 0.1,
    },
    "variants": {
        "sparse_sens": ({**_MAG_SPARSE}, "c5.2xlarge"),
        "sparse_dw2": ({**_MAG_SPARSE, "depth_weighting": "depth", "depth_weighting_exponent": 2.0},
                       "c5.2xlarge"),
        "sparse_dw3": ({**_MAG_SPARSE, "depth_weighting": "depth", "depth_weighting_exponent": 3.0},
                       "c5.2xlarge"),
        "l1l2": ({"regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "irls",
                  "l1l2_weighting": "S1", "beta_selection": "auto"}, "c5.2xlarge"),
    },
}
CASES = {"karnataka-gravity": KARNATAKA_GRAVITY, "synthetic-magnetic": SYNTHETIC_MAGNETIC}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def find_instance(backend, task_id):
    """Id and type of the live instance tagged with ``task_id``, or None."""
    res = backend.ec2.describe_instances(Filters=[
        {"Name": "tag:geoinv3d:task", "Values": [task_id]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    found = [i for r in res["Reservations"] for i in r["Instances"]]
    return (found[0]["InstanceId"], found[0]["InstanceType"]) if found else None


def run_variant(backend, name, case, params_extra, instance, stamp, out, launched,
                resume=False):
    task_id = f"{stamp}-{name}"
    if resume:
        found = find_instance(backend, task_id)
        if found is None:
            log(f"[{name}] no live instance tagged {task_id}")
            return
        job_id, instance = found
        launched.append(job_id)
        log(f"[{name}] resuming {job_id} ({instance})")
        # "uploading" makes the backend check the instance: follow a started
        # job, or set it up again (from ~/.geoinv3d/staging/<task>)
        record = {"job_id": job_id, "task_id": task_id, **backend.initial_fields(),
                  "phase": "uploading"}
    else:
        params = {**json.loads(json.dumps(case["base"])), **params_extra, "task_id": task_id}
        job_id = backend.start_job(task_id, [str(f) for f in case["files"]], params, instance)
        launched.append(job_id)
        log(f"[{name}] launched {job_id} ({instance})")
        record = {"job_id": job_id, "task_id": task_id, **backend.initial_fields()}
    t0, last, last_error = time.time(), None, None
    while True:
        record.update(backend.refresh(record))
        p = record.get("progress") or {}
        line = f"{record['phase']} {p.get('stage', '')}" + \
            (f" it {p['iteration']}/{p.get('max_iter')} phi_d {p.get('phi_d', 0):.4g}"
             f" (target {p.get('phi_d_target')})" if p.get("iteration") is not None else "")
        if line != last:
            log(f"[{name}] {line}")
            last = line
        error = record.get("refresh_error")
        if error and error != last_error:
            log(f"[{name}] ! {error}")
        last_error = error
        if record["phase"] in TERMINAL_PHASES or record["phase"] == "stopped":
            break
        time.sleep(15)
    minutes = (time.time() - t0) / 60
    summary = backend.result_summary(record) or {}
    out[name] = {"record": record, "summary": summary, "minutes": minutes,
                 "cost": PRICE.get(instance, 0) * minutes / 60, "instance": instance}
    log(f"[{name}] {record['phase']} after {minutes:.1f} min"
        + (f": {record.get('reason')}" if record.get("reason") else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--only", default="sparse,l1l2", help="comma-separated variants")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--resume", metavar="STAMP",
                    help="follow the instances of an earlier call instead of launching")
    args = ap.parse_args()

    case = CASES[args.case]
    names = [n.strip() for n in args.only.split(",")]
    backend = EC2Backend(args.region)
    stamp = args.resume or f"{args.case}-{int(time.time())}"
    log(f"{args.case}: {names}; owner {backend.owner}, region {args.region}")
    out, launched, threads = {}, [], []
    try:
        for name in names:
            extra, instance = case["variants"][name]
            t = threading.Thread(target=run_variant, daemon=True,
                                 args=(backend, name, case, extra, instance, stamp, out, launched,
                                       bool(args.resume)))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        for job_id in launched:
            try:
                backend.ec2.terminate_instances(InstanceIds=[job_id])
            except Exception as e:
                log(f"could not terminate {job_id}: {e}")
        log(f"terminated {launched} (safety net; harmless if already terminated)")

    log("summary:")
    ok = []
    for name in names:
        r = out.get(name)
        if not r:
            log(f"  {name}: no result")
            continue
        s = r["summary"]
        its = s.get("iterations") or [{}]
        nd = s.get("n_data") or 1
        log(f"  {name:7s} {r['record']['phase']:9s} {r['minutes']:5.1f} min ~${r['cost']:.2f} "
            f"{s.get('n_iterations')} it, chi2/N {its[-1].get('phi_d', float('nan')) / nd:.2f}, "
            f"regional {((s.get('datasets') or [{}])[0].get('regional') or {}).get('order')}, "
            f"notes {s.get('notes')}")
        if r["record"]["phase"] == "succeeded":
            ok.append(backend.result_dir(r["record"]))
    if ok:
        from geoinv3d.viz.result_workflow import build_workflow, load_result
        from geoinv3d.viz.serve_dag import generate_viewer
        runs = [load_result(d) for d in ok]
        for run, d in zip(runs, ok):
            run["_name"] = d.name.split("-")[-1]
        dest = REPO / "examples" / "output" / "ec2_runs" / f"{stamp}.geoinv3d.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
        log(f"viewer: {generate_viewer(str(dest))}")
    return 0 if len(ok) == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
