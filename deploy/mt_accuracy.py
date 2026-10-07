"""MT forward accuracy: the designed meshes against the analytic impedance of layered earths.

    python deploy/mt_accuracy.py                  # on one EC2 instance (PARDISO), then terminate it
    python deploy/mt_accuracy.py --local          # here (slow with SciPy's SuperLU)
    python deploy/mt_accuracy.py --full           # 9 frequencies, tensor and OcTree, and the uniform
                                                  # and Bostick primaries too (these fail at 0.01 Hz)

For each model (a halfspace and three layered earths) the stations' apparent resistivity sets
the mesh (geoinv3d.cloud.meshing.recommend_mt_mesh) and the primary (worker._mt_primaries:
the smoothest layering fitting it), as a job would; the impedance at three stations is
compared with the analytic 1D impedance of the model as the cells hold it (each cell under
the middle station a layer).  It passes when every apparent resistivity is within
``--max-rho`` % and every phase within ``--max-phase`` degrees (exit code 0).

The data are a 10 km array at 0.01-100 Hz, two frequencies a decade (they set the mesh and the
primary, as a job's data do); the quick suite (the default) solves the lowest, a middle and the
highest of them: the lowest tests the padding and the primary, the highest the top cells.  About
15 minutes on an m5.4xlarge (~$0.2).  The primary needs that sampling: from three frequencies two
decades apart, the deep layering is a guess (10 over 1000 ohm m: 12 % off at 0.01 Hz).
LOGBOOK, 2026-10-07: the study behind it.

On EC2 the instance is tagged Owner=<IAM user> like the jobs, its boot-time shutdown is
replaced by one at ``--timeout-min``, and it is terminated at the end, also after errors.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = {"halfspace 100": ([100.0], []), "10 over 1000 at 300 m": ([10.0, 1000.0], [300.0]),
          "1000 over 10 at 1 km": ([1000.0, 10.0], [1000.0]),
          "100/10/1000": ([100.0, 10.0, 1000.0], [500.0, 2000.0])}
EXTENT, SPACING = (0.0, 10000.0, 0.0, 10000.0), 2000.0


def _flat(x, y):
    return np.zeros_like(np.asarray(x, dtype=float))


DATA_FREQUENCIES = np.logspace(-2, 2, 9)


def run_suite(freqs, kinds=("octree",), primaries=("smooth1d",), max_rho=3.0, max_phase=1.5,
              data_freqs=DATA_FREQUENCIES) -> bool:
    """Print a line per (mesh, primary, model): the mesh and the primary from the data at
    ``data_freqs``, the impedance solved at ``freqs``; True when all are within the limits."""
    from geoinv3d.cloud.meshing import (MU0, _rho_a_1d, bostick_layers, recommend_mt_mesh,
                                        smooth_1d_layers)
    from geoinv3d.cloud.worker import (_active_below_surface, _build_mt_octree_mesh,
                                       _build_mt_tensor_mesh)
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.mt import MTMethod
    from geoinv3d.methods.solvers import pde_solver
    warnings.filterwarnings("ignore")
    freqs, data_freqs = np.asarray(freqs, float), np.asarray(data_freqs, float)
    print(f"solver {pde_solver()[1]}; data at {data_freqs.size} frequencies "
          f"{data_freqs.min():g}-{data_freqs.max():g} Hz, solved at {np.round(freqs, 4).tolist()} Hz", flush=True)
    ok = True
    xs = np.linspace(EXTENT[0], EXTENT[1], 3)
    st = np.column_stack([xs, np.full(3, 0.5 * (EXTENT[2] + EXTENT[3])), np.zeros(3)])
    comps = ["xy_real", "xy_imag", "yx_real", "yx_imag"]
    for kind in kinds:
        for name, (rho, thick) in MODELS.items():
            ra = _rho_a_1d(data_freqs, rho, thick)
            stats = {"frequencies": list(data_freqs), "p10": list(ra), "p50": list(ra), "p90": list(ra)}
            design = recommend_mt_mesh(EXTENT, SPACING, data_freqs, stats)
            build = _build_mt_octree_mesh if kind == "octree" else _build_mt_tensor_mesh
            mesh = build(EXTENT, _flat, False, design)
            dm = mesh.to_discretize()
            active = _active_below_surface(dm, _flat)
            tops = np.r_[0.0, -np.cumsum(thick)]
            cc = dm.cell_centers[active]
            m = np.full(active.sum(), np.log(1 / rho[-1]))
            for j in range(len(rho) - 1):
                m[(cc[:, 2] < tops[j]) & (cc[:, 2] >= tops[j + 1])] = np.log(1 / rho[j])
            # the cells' own layering under the middle station
            nodes = dm.origin[2] + np.r_[0.0, np.cumsum(dm.h[2])]
            zc = 0.5 * (nodes[1:] + nodes[:-1])
            g = zc < 0
            rc = np.full(g.sum(), rho[-1])
            for j in range(len(rho) - 1):
                rc[(zc[g] < tops[j]) & (zc[g] >= tops[j + 1])] = rho[j]
            rc, hc = rc[::-1], dm.h[2][g][::-1]
            w = 2 * np.pi * freqs[:, None]
            k = np.sqrt(1j * w * MU0 / rc[None, :])
            eta = 1j * w * MU0 / k
            za = eta[:, -1]
            for j in range(rc.size - 2, -1, -1):
                t = np.tanh(k[:, j] * hc[j])
                za = eta[:, j] * (za + eta[:, j] * t) / (eta[:, j] + za * t)
            for prim in primaries:
                kw = {}
                if prim != "uniform":
                    kw["primary_layers"] = (smooth_1d_layers if prim == "smooth1d" else bostick_layers)(stats)
                mt = MTMethod(frequencies=freqs, components=comps,
                              sigma_background=1 / np.exp(np.mean(np.log(ra))), **kw)
                n = len(freqs) * len(comps) * 3
                sv = SurveyData(locations=st, observed=np.zeros(n), std=np.ones(n), method="mt")
                t0 = time.time()
                d = mt.make_simulation_active(mesh, sv, active, forward_only=True).dpred(m)
                d = d.reshape(len(freqs), len(comps), 3)
                zxy, zyx = d[:, 0] + 1j * d[:, 1], d[:, 2] + 1j * d[:, 3]
                er = np.r_[[100 * (np.abs(z) ** 2 / np.abs(za[:, None]) ** 2 - 1) for z in (zxy, zyx)]]
                ph = np.r_[[(np.degrees(np.angle(z / (za[:, None] * s))) + 180) % 360 - 180
                            for z, s in ((zxy, -1), (zyx, 1))]]
                worst_r, worst_p = np.abs(er).max(), np.abs(ph).max()
                good = worst_r <= max_rho and worst_p <= max_phase
                ok &= good
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
                print(f"{'ok  ' if good else 'FAIL'} {kind:6s} {prim:8s} {name:22s} {dm.n_cells:7d} cells "
                      f"{time.time() - t0:5.0f} s | rho_a {worst_r:5.2f} % phase {worst_p:4.2f} deg "
                      f"| per f {np.round(np.abs(er).max(axis=(0, 2)), 1).tolist()} | RSS {rss:.1f} GB", flush=True)
    print("PASSED" if ok else "FAILED", flush=True)
    return ok


def on_ec2(args) -> int:
    sys.path.insert(0, str(REPO / "deploy"))
    import ec2_sweep as es
    from geoinv3d.cloud.ec2 import REMOTE, EC2Backend, code_archive
    b = EC2Backend(args.region)
    iid = es.launch(b, f"mtaccuracy-{int(time.time())}", 0, args.type)
    print(time.strftime("%H:%M:%S"), "launched", iid, args.type, "owner", b.owner, flush=True)
    try:
        ssh = es.ssh_to(b, iid)
        while ssh.read(f"{REMOTE}/READY") is None:
            if ssh.read(f"{REMOTE}/BOOTSTRAP_FAILED") is not None:
                print(ssh.run("sudo tail -n 30 /var/log/geoinv3d-bootstrap.log")[1])
                return 2
            time.sleep(10)
        # the boot-time shutdown waits for a job; this is none: our own limit instead
        ssh.run(f"sudo shutdown -c; sudo shutdown -h +{args.timeout_min}")
        ssh.write(f"{REMOTE}/code.tar.gz", code_archive())
        ssh.run(f"rm -rf {REMOTE}/code/geoinv3d && tar xzf {REMOTE}/code.tar.gz -C {REMOTE}/code")
        ssh.write(f"{REMOTE}/mt_accuracy.py", Path(__file__).read_bytes())
        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in
                         (("max_rho", args.max_rho), ("max_phase", args.max_phase)))
        inner = (f"cd {REMOTE} && PYTHONPATH={REMOTE}/code {REMOTE}/venv/bin/python -u mt_accuracy.py "
                 f"--local {'--full ' if args.full else ''}{flags} > accuracy.log 2>&1")
        ssh.run(f"{{ nohup setsid bash -c '{inner}' > /dev/null 2>&1 < /dev/null & }} && echo started")
        shown = 0
        while True:
            time.sleep(30)
            log = (ssh.read(f"{REMOTE}/accuracy.log") or b"").decode(errors="replace").splitlines()
            lines = [ln for ln in log if "Warning" not in ln and "warn" not in ln and ln.strip()]
            for ln in lines[shown:]:
                print(ln, flush=True)
            shown = len(lines)
            if any(ln in ("PASSED", "FAILED") for ln in lines) or any("Traceback" in ln for ln in lines):
                return 0 if "PASSED" in lines else 1
    finally:
        b.ec2.terminate_instances(InstanceIds=[iid])
        print(time.strftime("%H:%M:%S"), "terminated", iid, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help="run here instead of on EC2")
    ap.add_argument("--full", action="store_true", help="9 frequencies, tensor and OcTree, all primaries")
    ap.add_argument("--max-rho", type=float, default=3.0)
    ap.add_argument("--max-phase", type=float, default=1.5)
    ap.add_argument("--type", default="m5.4xlarge")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--timeout-min", type=int, default=120)
    args = ap.parse_args()
    if not args.local:
        return on_ec2(args)
    if args.full:
        ok = run_suite(DATA_FREQUENCIES, kinds=("octree", "tensor"),
                       primaries=("uniform", "bostick", "smooth1d"), max_rho=args.max_rho, max_phase=args.max_phase)
    else:
        ok = run_suite([0.01, 1.0, 100.0], max_rho=args.max_rho, max_phase=args.max_phase)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
