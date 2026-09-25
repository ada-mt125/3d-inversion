"""Example: submit an inversion to AWS Batch.

Shows the full cloud workflow:
  1. Build an InversionTask from local data
  2. Pack it into a portable archive
  3. (Optional) Submit to AWS Batch and wait for results
  4. Locally execute the same task to verify

Usage:
    python examples/cloud_inversion.py          # local test
    python examples/cloud_inversion.py --aws    # submit to AWS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geoinv3d.cloud import InversionTask, pack_task, unpack_task, AWSRunner
from geoinv3d.cloud.worker import execute_task, pack_result


def build_synthetic_gravity_task() -> InversionTask:
    """Create a gravity inversion task with synthetic data."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.methods.gravity import GravityMethod

    np.random.seed(42)

    nx, ny, nz = 10, 10, 5
    dx, dy, dz = 100.0, 100.0, 50.0
    mesh = Mesh3D.uniform(nx, ny, nz, dx, dy, dz)

    true_model = np.zeros(mesh.n_cells)
    d3d = true_model.reshape(nx, ny, nz)
    d3d[3:7, 3:7, 1:3] = 0.3

    xs = np.linspace(150, 850, 6)
    ys = np.linspace(150, 850, 6)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(36)])

    from geoinv3d.datamodel.survey import SurveyData
    survey = SurveyData(locs, np.zeros(36), np.ones(36), method="gravity")
    method = GravityMethod()
    sim = method.make_simulation_full(mesh, survey)
    dpred = sim.dpred(true_model)

    noise = 0.02 * np.abs(dpred) * np.random.randn(36)
    dobs = dpred + noise
    std = 0.02 * np.abs(dpred) + 1e-6

    return InversionTask(
        task_id="gravity_cloud_test",
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        method_type="gravity",
        regularization_type="sparse",
        norms=(0.0, 2.0, 2.0, 1.0),
        max_iter=15,
        beta0_ratio=1.0,
        irls_cooling_factor=1.1,
        max_irls_iterations=12,
        use_preconditioner=True,
        initial_model=np.zeros(mesh.n_cells),
        observed_data=dobs,
        data_std=std,
        station_locations=locs,
    )


def main():
    use_aws = "--aws" in sys.argv
    bucket = os.environ.get("GEOINV3D_BUCKET", "geoinv3d-tasks")

    print("=" * 60)
    print("GeoInv3D Cloud Inversion Example")
    print("=" * 60)

    print("\n[1] Building synthetic gravity task...")
    task = build_synthetic_gravity_task()
    print(f"    Task ID: {task.task_id}")
    print(f"    Mesh: {task.nx}x{task.ny}x{task.nz}")
    print(f"    Stations: {len(task.station_locations)}")
    print(f"    Max iterations: {task.max_iter}")

    print("\n[2] Packing task archive...")
    archive = pack_task(task, f"{task.task_id}.geoinv3d.zip")
    size_kb = os.path.getsize(archive) / 1024
    print(f"    Archive: {archive} ({size_kb:.1f} KB)")

    print("\n[3] Verifying round-trip...")
    task2 = unpack_task(archive)
    assert task2.task_id == task.task_id
    assert task2.nx == task.nx
    assert np.allclose(task2.observed_data, task.observed_data)
    print("    Round-trip OK")

    if use_aws:
        print("\n[4] Submitting to AWS Batch...")
        runner = AWSRunner(bucket=bucket)
        status = runner.submit_and_wait(task, poll_interval=15)
        print(f"    Status: {status['status']}")

        if status["status"] == "SUCCEEDED":
            print(f"    Result: {status['result_path']}")
        else:
            print(f"    Reason: {status.get('reason', 'unknown')}")
    else:
        print("\n[4] Running inversion locally (same as cloud worker)...")
        result = execute_task(task)
        print(f"    Method: {result['method']}")
        print(f"    Iterations: {result['n_iterations']}")

        if result.get("iterations"):
            last = result["iterations"][-1]
            print(f"    Final phi_d: {last['phi_d']:.2f}")
            print(f"    Final phi_m: {last['phi_m']:.6f}")
            print(f"    Model range: [{last['model_min']:.4f}, "
                  f"{last['model_max']:.4f}]")

        result_path = pack_result(result, f"{task.task_id}_result.zip")
        size_kb = os.path.getsize(result_path) / 1024
        print(f"    Result archive: {result_path} ({size_kb:.1f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
