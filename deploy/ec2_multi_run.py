"""Run one dataset with several regularizations in parallel on EC2, then compare.

    python deploy/ec2_multi_run.py karnataka-gravity [--only sparse,l1l2]
    python deploy/ec2_multi_run.py karnataka-gravity --resume karnataka-gravity-1790537566
    python deploy/ec2_multi_run.py karnataka-joint --only none --local --collect DIR

``--local`` runs the variants on this machine, one after the other, instead of on EC2 (no AWS
account needed; a full-resolution Karnataka run needs about 16 GB of memory), and writes each
result to DIR/<variant>/ like ``--collect``.

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

PRICE = {"c5.xlarge": 0.17, "c5.2xlarge": 0.34, "c5.4xlarge": 0.68, "c5.9xlarge": 1.53,
         "c5.12xlarge": 2.04, "c5.18xlarge": 3.06}
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

# Karnataka with terrain (examples/output/karnataka_inputs/prepare_inputs.py): the ground from
# the Copernicus DEM, 250 m layers so that the relief (380-1090 m) is more than two steps, the
# Bouguer grid with the terrain correction added, TMI continued upwards to 1 km above the ground
_KI = REPO / "examples" / "output" / "karnataka_inputs"
_NGPM = DESKTOP / "karnataka_ap_gravity" / "GEOTIFF" / "NGPM_BA.tiff"
_TERRAIN = {
    "crs": "EPSG:32643", "topography": {"file": "dem_utm43n_450m.tif"}, "mesh_type": "tensor",
    "core_cell_m": 1000.0, "core_cell_z_m": 250.0, "depth_core_m": 10000.0, "pad_distance_m": 20000.0,
    "param_mode": "manual", "beta0_ratio": 1.0, "cooling_factor": 2.0, "use_preconditioner": True,
}
_SPARSE = {"regularization_type": "sparse", "norms": [0, 2, 2, 2], "alpha_s": 1.0,
           "alpha_x": 1, "alpha_y": 1, "alpha_z": 1}


def _dw(beta):
    return {"depth_weighting": "depth", "depth_weighting_exponent": float(beta)}


def _gravity_data(name, **extra):
    return [{"method": "gravity", "files": [name], "component": "gz", "noise_pct": 0.0,
             "noise_floor": 0.5, "regional": {"method": "polynomial", "order": 2}, **extra}]


_L1L2 = {"regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "irls",
         "l1l2_weighting": "S1", "beta_selection": "auto"}
KARNATAKA_GRAVITY_TERRAIN = {
    "files": [_KI / "gravity_complete_1km.csv", _KI / "gravity_simple_1km.csv",
              _KI / "dem_utm43n_450m.tif", _NGPM],
    "base": {
        **_TERRAIN, "method_type": "gravity", "inversion_mode": "single",
        "datasets": _gravity_data("gravity_complete_1km.csv"),
        "max_iter": 30, "max_irls_iterations": 30, "bounds_lower": -0.2, "bounds_upper": 0.5,
    },
    "variants": {
        # the six settings of the flat-earth comparison (28 September)
        "original_sparse": ({**_SPARSE, "norms": [0, 2, 2, 1], "alpha_s": 1e-4}, "c5.4xlarge"),
        "l1l2_irls": (_L1L2, "c5.4xlarge"),
        "as1_beta0.5": ({**_SPARSE, **_dw(0.5)}, "c5.4xlarge"),
        "as1_beta1": ({**_SPARSE, **_dw(1.0)}, "c5.4xlarge"),
        "as1_beta1.5": ({**_SPARSE, **_dw(1.5)}, "c5.4xlarge"),
        "as0.1_beta1": ({**_SPARSE, "alpha_s": 0.1, **_dw(1.0)}, "c5.4xlarge"),
        # what each change does, at beta = 1: the terrain without the terrain correction,
        # and the flat earth of the earlier runs on the 250 m layers of this mesh
        "as1_beta1_no_tc": ({**_SPARSE, **_dw(1.0),
                             "datasets": _gravity_data("gravity_simple_1km.csv")}, "c5.4xlarge"),
        "as1_beta1_flat": ({**_SPARSE, **_dw(1.0), "datasets": _gravity_data("NGPM_BA.tiff"),
                            "aoi": [641000, 711000, 1634000, 1704000], "decimate_stride": 2,
                            "topography": {"flat_elevation": 0}}, "c5.4xlarge"),
    },
}

_KARNATAKA_FIELD = [42100.0, 19.3, -1.4]      # IGRF 2020 at the centre; declination to grid north


def _magnetic_data(noise_pct=0.05, noise_floor=10.0, name="magnetic_1km.csv", **extra):
    return [{"method": "magnetics", "files": [name], "component": "tmi",
             "method_kwargs": {"inducing_field": _KARNATAKA_FIELD}, "noise_pct": noise_pct,
             "noise_floor": noise_floor, "regional": {"method": "polynomial", "order": 2},
             **extra}]


KARNATAKA_MAGNETIC = {
    "files": [_KI / "magnetic_1km.csv", _KI / "magnetic_1km_h500.csv", _KI / "magnetic_1km_raw80.csv",
              _KI / "magnetic_1km_xy.csv", _KI / "dem_utm43n_450m.tif"],
    "base": {
        **_TERRAIN, "method_type": "magnetics", "inversion_mode": "single",
        "datasets": _magnetic_data(), "method_kwargs": {"inducing_field": _KARNATAKA_FIELD},
        "max_iter": 60, "max_irls_iterations": 40, "bounds_lower": 0.0, "bounds_upper": 1.0,
    },
    "variants": {
        "sens": ({**_SPARSE}, "c5.4xlarge"),
        "beta1": ({**_SPARSE, **_dw(1.0)}, "c5.4xlarge"),
        "beta1.5": ({**_SPARSE, **_dw(1.5)}, "c5.4xlarge"),
        "beta2": ({**_SPARSE, **_dw(2.0)}, "c5.4xlarge"),
        "beta3": ({**_SPARSE, **_dw(3.0)}, "c5.4xlarge"),
        "l1l2_irls": (_L1L2, "c5.4xlarge"),
        "beta0.5": ({**_SPARSE, **_dw(0.5)}, "c5.4xlarge"),
        "as0.1_beta1": ({**_SPARSE, "alpha_s": 0.1, **_dw(1.0)}, "c5.4xlarge"),
        # tests at beta = 1: the upper bound, the data error, the height the data are continued
        # to (500 m; 80 m = as flown, aliased at 1 km) and a flat earth
        "beta1_ub0.3": ({**_SPARSE, **_dw(1.0), "bounds_upper": 0.3}, "c5.4xlarge"),
        # remanence: a magnetization vector per cell (MVI; three components, so about 20 GB
        # of sensitivities), |m_i| <= 1 SI, otherwise as beta1
        "beta1_mvi": ({**_SPARSE, **_dw(1.0),
                       "datasets": _magnetic_data(method_kwargs={"inducing_field": _KARNATAKA_FIELD,
                                                                 "magnetization": "vector"}),
                       "method_kwargs": {"inducing_field": _KARNATAKA_FIELD, "magnetization": "vector"}},
                      "c5.9xlarge"),
        "beta1_err2": ({**_SPARSE, **_dw(1.0), "datasets": _magnetic_data(0.02, 5.0)}, "c5.4xlarge"),
        "beta1_h500": ({**_SPARSE, **_dw(1.0),
                        "datasets": _magnetic_data(name="magnetic_1km_h500.csv")}, "c5.4xlarge"),
        "beta1_raw80": ({**_SPARSE, **_dw(1.0),
                         "datasets": _magnetic_data(name="magnetic_1km_raw80.csv")}, "c5.4xlarge"),
        "beta1_flat": ({**_SPARSE, **_dw(1.0), "topography": {"flat_elevation": 0},
                        "datasets": _magnetic_data(name="magnetic_1km_xy.csv", station_height=1000.0)},
                       "c5.4xlarge"),
    },
}

# Joint gravity + magnetics over the same area and mesh: one run per coupling (parameters in
# examples/output/karnataka_joint/scripts/joint_params.py, shared with the local 2 km study)
sys.path.insert(0, str(REPO / "examples" / "output" / "karnataka_joint" / "scripts"))
from joint_params import COUPLINGS as _JOINT_COUPLINGS, VARIANTS as _JOINT_VARIANTS   # noqa: E402
from joint_params import base as _joint_base   # noqa: E402

KARNATAKA_JOINT = {
    "files": [_KI / "gravity_complete_1km.csv", _KI / "magnetic_1km.csv", _KI / "dem_utm43n_450m.tif"],
    "base": _joint_base(),
    "variants": {name: (extra, "c5.18xlarge" if name.startswith("group_lasso") else "c5.9xlarge")
                 for name, extra in _JOINT_COUPLINGS.items()},
}
# no coupling and JTV with the models' regularization of a joint_params.VARIANTS entry (the
# density bounds of the rock samples, the depth weighting): e.g. "joint_total_variation_rho35_gb05"
KARNATAKA_JOINT["variants"].update({
    f"{c}_{key}": ({**_JOINT_COUPLINGS[c], "datasets": _joint_base(**v)["datasets"]}, "c5.9xlarge")
    for c in ("none", "joint_total_variation") for key, v in _JOINT_VARIANTS.items()})
CASES = {"karnataka-gravity": KARNATAKA_GRAVITY, "synthetic-magnetic": SYNTHETIC_MAGNETIC,
         "karnataka-gravity-terrain": KARNATAKA_GRAVITY_TERRAIN,
         "karnataka-magnetic": KARNATAKA_MAGNETIC, "karnataka-joint": KARNATAKA_JOINT}


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
                resume=False, slots=None):
    with slots or threading.Semaphore():
        _run_variant(backend, name, case, params_extra, instance, stamp, out, launched, resume)


def _run_variant(backend, name, case, params_extra, instance, stamp, out, launched, resume):
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
        while True:    # at the vCPU limit of the account: wait for another run to end
            try:
                job_id = backend.start_job(task_id, [str(f) for f in case["files"]], params, instance)
                break
            except Exception as e:
                if "VcpuLimitExceeded" not in str(e) and "InsufficientInstanceCapacity" not in str(e):
                    raise
                log(f"[{name}] waiting for capacity: {str(e)[:90]}")
                time.sleep(60)
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


def run_local(case, names, stamp, collect) -> int:
    """The variants on this machine, in turn; results in collect/<variant>/."""
    import shutil
    import tempfile
    from geoinv3d.cloud.worker import pack_result, result_metadata_json, run_data_pipeline

    data = Path(tempfile.mkdtemp(prefix="geoinv3d-"))
    for f in case["files"]:
        shutil.copy(f, data / Path(f).name)
    failed = 0
    for name in names:
        extra, _ = case["variants"][name]
        params = {**json.loads(json.dumps(case["base"])), **extra, "task_id": f"{stamp}-{name}"}
        dest = Path(collect) / name
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log(f"[{name}] running locally")
        try:
            result = run_data_pipeline(params, str(data))
        except Exception as e:
            failed += 1
            log(f"[{name}] failed: {e}")
            continue
        pack_result(result, str(dest / "result.zip"))
        (dest / "result.json").write_text(result_metadata_json(result), encoding="utf-8")
        minutes = (time.time() - t0) / 60
        (dest / "run.json").write_text(json.dumps(
            {"task_id": params["task_id"], "instance": "local", "minutes": round(minutes, 1), "cost_usd": 0.0}))
        log(f"[{name}] done after {minutes:.1f} min -> {dest}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--only", default="sparse,l1l2", help="comma-separated variants")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--resume", metavar="STAMP",
                    help="follow the instances of an earlier call instead of launching")
    ap.add_argument("--parallel", type=int, default=0, help="instances at a time (default: all)")
    ap.add_argument("--collect", metavar="DIR",
                    help="copy each result (result.zip, result.json) to DIR/<variant>/")
    ap.add_argument("--local", action="store_true",
                    help="run on this machine instead of EC2 (needs --collect)")
    args = ap.parse_args()

    case = CASES[args.case]
    names = [n.strip() for n in args.only.split(",")]
    if args.local:
        if not args.collect:
            ap.error("--local needs --collect DIR")
        return run_local(case, names, f"{args.case}-local", args.collect)
    backend = EC2Backend(args.region)
    stamp = args.resume or f"{args.case}-{int(time.time())}"
    log(f"{args.case}: {names}; owner {backend.owner}, region {args.region}")
    out, launched, threads = {}, [], []
    slots = threading.Semaphore(args.parallel or len(names))
    try:
        for name in names:
            extra, instance = case["variants"][name]
            t = threading.Thread(target=run_variant, daemon=True,
                                 args=(backend, name, case, extra, instance, stamp, out, launched,
                                       bool(args.resume), slots))
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
            if args.collect:
                import shutil
                dest = Path(args.collect) / name
                dest.mkdir(parents=True, exist_ok=True)
                for fname in ("result.zip", "result.json"):
                    shutil.copy(ok[-1] / fname, dest / fname)
                (dest / "run.json").write_text(json.dumps(
                    {"task_id": r["record"]["task_id"], "instance": r["instance"],
                     "minutes": round(r["minutes"], 1), "cost_usd": round(r["cost"], 3)}))
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
