"""End-to-end AWS pipeline: upload raw data -> run inversion -> get results.

Shows both workflows:
  1. Pre-packed sparse IRLS task (for prepared data)
  2. Raw data pipeline (upload GeoTIFF/CSV directly to AWS)

Usage:
    # Local test (no AWS needed):
    python examples/aws_pipeline.py

    # Submit to AWS with pre-packed task:
    python examples/aws_pipeline.py --aws

    # Full pipeline: upload raw data + run on AWS:
    python examples/aws_pipeline.py --aws --pipeline /path/to/gravity.tif

    # With custom parameters:
    python examples/aws_pipeline.py --aws --pipeline /path/to/data.tif \
        --method gravity --norms 0,2,2,1 --depth 8000 --cell 500
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np

from geoinv3d.cloud import InversionTask, pack_task, unpack_task, AWSRunner
from geoinv3d.cloud.worker import execute_task, pack_result


def build_sparse_gravity_task() -> InversionTask:
    """Create a gravity inversion task with sparse IRLS settings."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.datamodel.survey import SurveyData

    np.random.seed(42)

    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 50.0, 50.0, 25.0

    n_pad = 5
    pad_factor = 1.3

    def _padded_h(n_core, dh, n_pad, factor):
        core = np.full(n_core, dh)
        pad = dh * factor ** np.arange(1, n_pad + 1)
        return np.concatenate([pad[::-1], core, pad])

    hx = _padded_h(nx, dx, n_pad, pad_factor)
    hy = _padded_h(ny, dy, n_pad, pad_factor)
    core_z = np.full(nz, dz)
    pad_z = dz * pad_factor ** np.arange(1, n_pad + 1)
    hz = np.concatenate([pad_z[::-1], core_z])

    mesh = Mesh3D(hx=hx, hy=hy, hz=hz, origin=(0, 0, -np.sum(hz)))

    true_model = np.zeros(mesh.n_cells)
    dmesh = mesh.to_discretize()
    cc = dmesh.cell_centers
    cx, cy, cz = 500, 500, -150
    dist = np.sqrt((cc[:, 0] - cx)**2 + (cc[:, 1] - cy)**2 + (cc[:, 2] - cz)**2)
    true_model[dist < 200] = 0.4

    xs = np.linspace(100, 900, 12)
    ys = np.linspace(100, 900, 12)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(144)])

    survey = SurveyData(locs, np.zeros(144), np.ones(144), method="gravity")
    method = GravityMethod()
    sim = method.make_simulation_full(mesh, survey)
    dpred = sim.dpred(true_model)

    noise = 0.02 * np.abs(dpred) * np.random.randn(144)
    dobs = dpred + noise
    std = 0.05 * np.abs(dpred) + 0.5

    return InversionTask(
        task_id="sparse_gravity_test",
        hx=hx, hy=hy, hz=hz,
        origin=(0, 0, -np.sum(hz)),
        method_type="gravity",
        regularization_type="sparse",
        norms=(0.0, 2.0, 2.0, 1.0),
        max_iter=25,
        irls_cooling_factor=1.1,
        max_irls_iterations=12,
        use_preconditioner=True,
        alpha_s=1e-3,
        noise_pct=0.05,
        noise_floor=0.5,
        initial_model=np.zeros(mesh.n_cells),
        observed_data=dobs,
        data_std=std,
        station_locations=locs,
    )


def run_local_test():
    """Local test with synthetic data — no AWS needed."""
    print("=" * 60)
    print("GeoInv3D Sparse IRLS Inversion (Local Test)")
    print("=" * 60)

    print("\n[1] Building sparse gravity task with padding cells...")
    task = build_sparse_gravity_task()
    print(f"    Task ID:      {task.task_id}")
    print(f"    Mesh:         hx={len(task.hx)} hy={len(task.hy)} hz={len(task.hz)}")
    print(f"    Stations:     {len(task.station_locations)}")
    print(f"    Reg type:     {task.regularization_type}")
    print(f"    Norms:        {task.norms}")
    print(f"    Preconditioner: {task.use_preconditioner}")
    print(f"    Max iter:     {task.max_iter}")

    print("\n[2] Packing/unpacking round-trip...")
    archive = pack_task(task, f"{task.task_id}.geoinv3d.zip")
    size_kb = os.path.getsize(archive) / 1024
    task2 = unpack_task(archive)
    assert task2.regularization_type == "sparse"
    assert tuple(task2.norms) == (0.0, 2.0, 2.0, 1.0)
    assert task2.use_preconditioner is True
    assert np.allclose(task2.hx, task.hx)
    print(f"    Archive: {archive} ({size_kb:.1f} KB) — round-trip OK")

    print("\n[3] Running sparse IRLS inversion locally...")
    result = execute_task(task)
    print(f"    Regularization: {result.get('regularization', 'unknown')}")
    print(f"    Norms used:     {result.get('norms', 'N/A')}")
    print(f"    Iterations:     {result['n_iterations']}")

    if result.get("iterations"):
        last = result["iterations"][-1]
        print(f"    Final phi_d:    {last['phi_d']:.2f}")
        print(f"    Model range:    [{last['model_min']:.4f}, {last['model_max']:.4f}]")

    m_rec = result["recovered_model"]
    print(f"\n    Recovered model: min={m_rec.min():.4f}, max={m_rec.max():.4f}")

    result_path = pack_result(result, f"{task.task_id}_result.zip")
    size_kb = os.path.getsize(result_path) / 1024
    print(f"    Result archive: {result_path} ({size_kb:.1f} KB)")

    os.unlink(archive)
    os.unlink(result_path)
    print("\nLocal test complete!")


def run_aws_task():
    """Submit pre-packed task to AWS."""
    bucket = os.environ.get("GEOINV3D_BUCKET", "geoinv3d-tasks")
    runner = AWSRunner(bucket=bucket)

    task = build_sparse_gravity_task()
    print(f"\n[AWS] Submitting sparse IRLS task (norms={task.norms})...")
    status = runner.submit_and_wait(task, poll_interval=15, timeout=7200)
    print(f"[AWS] Status: {status['status']}")
    if status["status"] == "SUCCEEDED":
        print(f"[AWS] Result: {status['result_path']}")


def run_aws_pipeline(data_path: str, args):
    """Upload raw data + run inversion pipeline on AWS."""
    bucket = os.environ.get("GEOINV3D_BUCKET", "geoinv3d-tasks")
    runner = AWSRunner(bucket=bucket)

    norms = [float(x) for x in args.norms.split(",")] if args.norms else [0, 2, 2, 1]

    params = {
        "method_type": args.method,
        "regularization_type": "sparse",
        "norms": norms,
        "core_cell_m": args.cell,
        "core_cell_z_m": args.cell / 2,
        "depth_core_m": args.depth,
        "pad_distance_m": 2000.0,
        "decimate_stride": args.stride,
        "max_iter": args.max_iter,
        "irls_cooling_factor": 1.1,
        "max_irls_iterations": 12,
        "use_preconditioner": True,
        "noise_pct": 0.05,
        "noise_floor": 0.5 if args.method == "gravity" else 1.0,
    }
    if args.method == "magnetics" and args.inducing:
        amp, inc, dec = [float(x) for x in args.inducing.split(",")]
        params["method_kwargs"] = {"inducing_field": (amp, inc, dec)}

    print(f"\n{'='*60}")
    print(f"GeoInv3D AWS Pipeline")
    print(f"{'='*60}")
    print(f"  Data:       {data_path}")
    print(f"  Method:     {args.method}")
    print(f"  Norms:      {norms}")
    print(f"  Cell size:  {args.cell}m (h) / {args.cell/2}m (v)")
    print(f"  Depth:      {args.depth}m")
    print(f"  Max iter:   {args.max_iter}")

    status = runner.run_pipeline(
        [data_path], params,
        poll_interval=30, timeout=14400,
    )
    print(f"\n[AWS] Status: {status['status']}")
    if status["status"] == "SUCCEEDED":
        print(f"[AWS] Result: {status['result_path']}")
    else:
        print(f"[AWS] Reason: {status.get('reason', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="GeoInv3D AWS Pipeline")
    parser.add_argument("--aws", action="store_true", help="Submit to AWS")
    parser.add_argument("--pipeline", type=str, help="Path to raw data file for pipeline mode")
    parser.add_argument("--method", default="gravity", choices=["gravity", "magnetics"])
    parser.add_argument("--norms", type=str, default="0,2,2,1", help="s,x,y,z norms")
    parser.add_argument("--cell", type=float, default=500.0, help="Horizontal cell size (m)")
    parser.add_argument("--depth", type=float, default=3000.0, help="Core mesh depth (m)")
    parser.add_argument("--stride", type=int, default=1, help="Data decimation stride")
    parser.add_argument("--max-iter", type=int, default=30, help="Max outer iterations")
    parser.add_argument("--inducing", type=str, help="Magnetic field: amp,inc,dec (nT,deg,deg)")
    args = parser.parse_args()

    if args.aws and args.pipeline:
        run_aws_pipeline(args.pipeline, args)
    elif args.aws:
        run_aws_task()
    else:
        run_local_test()


if __name__ == "__main__":
    main()
