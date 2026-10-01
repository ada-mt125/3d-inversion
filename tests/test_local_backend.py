"""The local backend: jobs on this machine, queued, followed through their files."""

import json
import sys
import time
import zipfile

import pytest

from geoinv3d.cloud.local import LocalBackend
from tests.test_data_pipeline import SMALL_MESH, _station_grid, _synthetic, _write_csv

# A stand-in for the worker: reports progress, waits for GO (or STOP), then writes its result
FAKE_WORKER = r"""
import json, os, sys, time
from pathlib import Path
job = Path(sys.argv[1]); out = job / "out"
(out / "progress.json").write_text(json.dumps({"stage": "inverting", "iteration": 1,
    "history": [{"i": 1, "phi_d": 100.0, "beta": 1.0}]}))
while not (job / "GO").exists() and not (out / "STOP").exists():
    time.sleep(0.02)
if (job / "CRASH").exists():
    os._exit(9)                      # dies without an exit code, as when out of memory
(out / "result.json").write_text(json.dumps({"n_iterations": 3, "stopped": (out / "STOP").exists()}))
(out / "result.zip").write_bytes(b"zip")
(out / "exit_code").write_text("0\n")
"""


@pytest.fixture
def backend(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    return LocalBackend(tmp_path / "local", max_jobs=1,
                        command=lambda job: [sys.executable, str(script), str(job)])


def _start(backend, tmp_path, task_id):
    data = tmp_path / f"{task_id}.csv"
    data.write_text("x,y,v\n0,0,1\n")
    job_id = backend.start_job(task_id, [str(data)], {"max_iter": 3}, None)
    return {"job_id": job_id, "task_id": task_id, **backend.initial_fields()}


def _wait(backend, record, phase, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        fields = backend.refresh(record)
        if fields["phase"] == phase:
            return fields
        time.sleep(0.05)
    raise AssertionError(f"{record['job_id']} did not reach {phase}: {fields}")


def _go(backend, record):
    (backend.job_dir(record) / "GO").write_text("")


class TestLocalBackend:
    def test_a_job_runs_and_its_result_is_collected(self, backend, tmp_path):
        rec = _start(backend, tmp_path, "a1")
        assert rec["job_id"] == "local-a1" and rec["backend"] == "local"
        job = backend.job_dir(rec)
        assert (job / "data" / "a1.csv").exists()
        assert json.loads((job / "params.json").read_text()) == {"max_iter": 3}
        running = _wait(backend, rec, "running")
        assert running["status"] == "RUNNING"
        t0 = time.time()
        while "history" not in running["progress"] and time.time() - t0 < 10:   # the report follows
            running = backend.refresh(rec)
        assert running["progress"]["history"][0]["phi_d"] == 100.0
        _go(backend, rec)
        done = _wait(backend, rec, "succeeded")
        assert done["status"] == "SUCCEEDED" and done["stopped"] >= done["started"]
        assert backend.result_summary(rec)["n_iterations"] == 3
        assert backend.fetch_result(rec).endswith("result.zip")

    def test_jobs_queue_one_at_a_time(self, backend, tmp_path):
        a, b = _start(backend, tmp_path, "q1"), _start(backend, tmp_path, "q2")
        _wait(backend, a, "running")
        queued = backend.refresh(b)
        assert queued["phase"] == "queued" and queued["status"] == "RUNNABLE"
        _go(backend, a)
        _wait(backend, a, "succeeded")
        _wait(backend, b, "running")      # started when the first ended
        _go(backend, b)
        _wait(backend, b, "succeeded")

    def test_cancel_a_running_and_a_queued_job(self, backend, tmp_path):
        a, b = _start(backend, tmp_path, "c1"), _start(backend, tmp_path, "c2")
        _wait(backend, a, "running")
        backend.cancel(b)
        assert backend.refresh(b)["phase"] == "cancelled"
        backend.cancel(a)
        fields = _wait(backend, a, "cancelled")
        assert fields["status"] == "FAILED" and fields["reason"] == "Cancelled by user"

    def test_stop_and_keep_the_result(self, backend, tmp_path):
        rec = _start(backend, tmp_path, "s1")
        with pytest.raises(RuntimeError):     # not started yet: nothing to stop
            LocalBackend(tmp_path / "other").request_finish(rec)
        _wait(backend, rec, "running")
        backend.request_finish(rec)
        _wait(backend, rec, "succeeded")
        assert backend.result_summary(rec)["stopped"] is True

    def test_a_worker_that_dies_without_a_result_failed(self, backend, tmp_path):
        rec = _start(backend, tmp_path, "d1")
        _wait(backend, rec, "running")
        (backend.job_dir(rec) / "CRASH").write_text("")
        _go(backend, rec)
        fields = _wait(backend, rec, "failed")
        assert "without a result" in fields["reason"]

    def test_jobs_outlive_a_restart_of_the_server(self, backend, tmp_path):
        rec = _start(backend, tmp_path, "r1")
        _wait(backend, rec, "running")
        again = LocalBackend(backend.root, command=backend.command)   # a new server process
        assert again.refresh(rec)["phase"] == "running"                # found by its pid
        _go(again, rec)
        _wait(again, rec, "succeeded")

    def test_resources(self, backend):
        r = backend.resources()
        assert r["cpus"] >= 1 and r["max_jobs"] == 1
        assert r["memory_gb"] is None or r["memory_gb"] > 0


def test_the_real_worker_runs_a_small_job(tmp_path):
    """The child process runs geoinv3d.cloud.worker on the job's files and writes exit_code."""
    locs = _station_grid(0.0)
    data = tmp_path / "g.csv"
    _write_csv(data, locs, _synthetic("gravity", locs))
    params = {"method_type": "gravity", "inversion_mode": "single",
              "datasets": [{"method": "gravity", "files": ["g.csv"], "noise_pct": 0.05,
                            "noise_floor": 0.01}],
              "param_mode": "manual", "regularization_type": "l2", "max_iter": 6,
              "topography": {"flat_elevation": 0}, "mesh_type": "tensor", **SMALL_MESH}
    backend = LocalBackend(tmp_path / "local", threads=2)
    rec = {"job_id": backend.start_job("w1", [str(data)], params), "task_id": "w1"}
    fields = _wait(backend, rec, "succeeded", timeout=180)
    assert fields["status"] == "SUCCEEDED"
    out = backend.job_dir(rec) / "out"
    assert (out / "exit_code").read_text().strip() == "0"
    progress = json.loads((out / "progress.json").read_text())
    assert progress["stage"] == "done" and len(progress["history"]) >= 2
    with zipfile.ZipFile(out / "result.zip") as zf:
        assert "recovered_model.npy" in zf.namelist()
    assert "convergence" in json.loads((out / "result.json").read_text())
