"""Job submission, monitoring and cancelling — with fake AWS clients only.

Nothing here talks to AWS: boto3 clients are replaced by in-memory fakes,
and ``boto3.client`` itself raises if anything tries to create a real one.
"""

import io
import json
import zipfile

import numpy as np
import pytest

import boto3

from geoinv3d.cloud.aws import AWSRunner, INSTANCE_RESOURCES, task_id_from_job_name
from geoinv3d.cloud import worker
from geoinv3d.cloud.worker import S3Progress
from tests.test_data_pipeline import SMALL_MESH, _station_grid, _synthetic, _write_csv


@pytest.fixture(autouse=True)
def no_real_aws(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to create a real boto3 client")
    monkeypatch.setattr(boto3, "client", refuse)


class NoSuchKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """Dict-backed S3: objects[key] = bytes."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []   # (key, body) in order

    def upload_file(self, local, bucket, key):
        self.objects[key] = open(local, "rb").read()

    def download_file(self, bucket, key, local):
        if key not in self.objects:
            raise NoSuchKey(key)
        open(local, "wb").write(self.objects[key])

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body
        self.puts.append((Key, Body))

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(fake.objects)
                                    if k.startswith(Prefix)]}
        return Paginator()


class FakeBatch:
    def __init__(self):
        self.calls = []
        self.jobs = {}

    def submit_job(self, **kwargs):
        self.calls.append(("submit_job", kwargs))
        return {"jobId": "job-123"}

    def describe_jobs(self, jobs):
        return {"jobs": [self.jobs[j] for j in jobs if j in self.jobs]}

    def terminate_job(self, jobId, reason):
        self.calls.append(("terminate_job", {"jobId": jobId, "reason": reason}))

    def cancel_job(self, jobId, reason):   # must not be used: it leaves running jobs alive
        self.calls.append(("cancel_job", {"jobId": jobId, "reason": reason}))


class FakeLogs:
    def get_log_events(self, **kwargs):
        self.kwargs = kwargs
        return {"events": [{"message": "[Worker] Starting"}, {"message": "iteration 3"}]}


def _runner():
    r = AWSRunner(bucket="b")
    r._s3, r._batch, r._logs = FakeS3(), FakeBatch(), FakeLogs()
    return r


# ── AWSRunner ───────────────────────────────────────────────────────────

class TestAWSRunner:
    def test_instance_type_sets_resource_requirements(self):
        r = _runner()
        r.submit_pipeline("t1", "geoinv3d/jobs/t1/data", {}, instance_type="c5.4xlarge")
        overrides = r.batch.calls[-1][1]["containerOverrides"]
        vcpus, memory = INSTANCE_RESOURCES["c5.4xlarge"]
        assert overrides["resourceRequirements"] == [
            {"type": "VCPU", "value": str(vcpus)}, {"type": "MEMORY", "value": str(memory)}]
        assert r.batch.calls[-1][1]["jobName"] == "geoinv3d-pipeline-t1"

        r.submit_pipeline("t2", "p", {})
        assert "resourceRequirements" not in r.batch.calls[-1][1]["containerOverrides"]
        with pytest.raises(ValueError, match="instance type"):
            r.submit_pipeline("t3", "p", {}, instance_type="m5.huge")

    def test_memory_fits_each_instance(self):
        ram_mib = {"c5.xlarge": 8192, "c5.2xlarge": 16384, "c5.4xlarge": 32768,
                   "c5.9xlarge": 73728}
        for name, (_, memory) in INSTANCE_RESOURCES.items():
            assert memory < ram_mib[name]

    def test_poll_reports_times_logs_and_task(self):
        r = _runner()
        r.batch.jobs["job-123"] = {
            "status": "RUNNING", "jobName": "geoinv3d-pipeline-abc", "createdAt": 1,
            "startedAt": 2, "container": {"logStreamName": "worker/x/1"}}
        s = r.poll("job-123")
        assert s == {"status": "RUNNING", "name": "geoinv3d-pipeline-abc", "task_id": "abc",
                     "created": 1, "started": 2, "log_stream": "worker/x/1"}
        assert r.poll("nope") == {"status": "UNKNOWN"}
        assert task_id_from_job_name("geoinv3d-t9") == "t9"

    def test_cancel_terminates_even_running_jobs(self):
        r = _runner()
        r.cancel("job-123", reason="Cancelled by user")
        assert r.batch.calls == [("terminate_job", {"jobId": "job-123",
                                                    "reason": "Cancelled by user"})]

    def test_progress_summary_and_logs(self):
        r = _runner()
        assert r.progress("t1") is None
        r.s3.objects["geoinv3d/jobs/t1/progress.json"] = b'{"stage": "inverting"}'
        r.s3.objects["geoinv3d/jobs/t1/result.json"] = b'{"n_iterations": 7}'
        assert r.progress("t1") == {"stage": "inverting"}
        assert r.result_summary("t1") == {"n_iterations": 7}
        assert r.tail_logs("worker/x/1", limit=2) == ["[Worker] Starting", "iteration 3"]
        assert r.logs.kwargs["logStreamName"] == "worker/x/1"


# ── Worker progress and exit status ─────────────────────────────────────

class TestS3Progress:
    def test_writes_on_stage_change_and_throttles_iterations(self):
        s3, now = FakeS3(), [100.0]
        p = S3Progress(s3, "b", "k/progress.json", min_interval=10, clock=lambda: now[0])
        p("inverting", max_iter=30)
        for it in range(1, 6):          # five iterations within 4 s: one write at most
            now[0] += 1
            p("inverting", iteration=it, phi_d=100.0 / it)
        now[0] += 10
        p("inverting", iteration=6, phi_d=10.0)
        p("done")
        stages = [json.loads(b) for _, b in s3.puts]
        assert [s["stage"] for s in stages] == ["inverting", "inverting", "done"]
        assert stages[1]["iteration"] == 6 and stages[1]["max_iter"] == 30

    def test_write_errors_do_not_raise(self, capsys):
        class Broken:
            def put_object(self, **kw):
                raise RuntimeError("no network")
        S3Progress(Broken(), "b", "k")("starting")
        assert "Could not update progress" in capsys.readouterr().out


def _worker_env(monkeypatch, s3, params):
    prefix = "geoinv3d/jobs/t1"
    s3.objects[f"{prefix}/params.json"] = json.dumps(params).encode()
    for k, v in {"TASK_BUCKET": "b", "TASK_ID": "t1", "RESULT_PREFIX": prefix,
                 "PIPELINE_MODE": "data", "PIPELINE_PARAMS": f"{prefix}/params.json",
                 "DATA_PREFIX": f"{prefix}/data"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(boto3, "client", lambda name, **kw: s3 if name == "s3" else None)
    return prefix


class TestWorkerMain:
    def test_success_reports_progress_and_uploads_results(self, monkeypatch, tmp_path):
        locs = _station_grid(0.0)
        _write_csv(tmp_path / "g.csv", locs, _synthetic("gravity", locs))
        s3 = FakeS3({"geoinv3d/jobs/t1/data/g.csv": (tmp_path / "g.csv").read_bytes()})
        params = {"method_type": "gravity", "inversion_mode": "single",
                  "datasets": [{"method": "gravity", "files": ["g.csv"], "noise_pct": 0.05,
                                "noise_floor": 0.01}],
                  "param_mode": "manual", "regularization_type": "l2", "max_iter": 6,
                  "topography": {"flat_elevation": 0}, "mesh_type": "tensor", **SMALL_MESH}
        prefix = _worker_env(monkeypatch, s3, params)
        worker.main()

        reports = [json.loads(b) for k, b in s3.puts if k.endswith("progress.json")]
        stages = [r["stage"] for r in reports]
        for stage in ("starting", "downloading", "loading_data", "building_mesh",
                      "inverting", "uploading", "done"):
            assert stage in stages
        inverting = [r for r in reports if r["stage"] == "inverting"]
        assert inverting[0]["max_iter"] == 6 and inverting[0]["phi_d_target"] == len(locs)
        summary = json.loads(s3.objects[f"{prefix}/result.json"])
        assert summary["n_iterations"] > 0 and "error" not in summary
        with zipfile.ZipFile(io.BytesIO(s3.objects[f"{prefix}/result.zip"])) as zf:
            assert "recovered_model.npy" in zf.namelist()

    def test_failure_uploads_the_error_and_exits_nonzero(self, monkeypatch):
        s3 = FakeS3()
        params = {"method_type": "gravity", "inversion_mode": "single",
                  "datasets": [{"method": "gravity", "files": ["missing.csv"]}]}
        prefix = _worker_env(monkeypatch, s3, params)
        with pytest.raises(SystemExit) as exc:
            worker.main()
        assert exc.value.code == 1   # Batch marks the job FAILED
        summary = json.loads(s3.objects[f"{prefix}/result.json"])
        assert "missing.csv" in summary["error"]
        last = json.loads([b for k, b in s3.puts if k.endswith("progress.json")][-1])
        assert last["stage"] == "failed" and "missing.csv" in last["message"]


# ── API server ──────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient   # noqa: E402

from geoinv3d.api import server   # noqa: E402


class FakeRunner:
    """Stands in for AWSRunner in the API: scripted Batch statuses."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.submitted, self.cancelled, self.polls = [], [], 0
        self.status = {"status": "RUNNABLE", "name": "geoinv3d-pipeline-t1", "task_id": "t1",
                       "created": 1000}
        self.progress_report = None
        self.summary = None

    def upload_data(self, paths, task_id):
        self.uploaded = [p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths]
        return "t1", "geoinv3d/jobs/t1/data"

    def submit_pipeline(self, task_id, data_prefix, params, instance_type=None):
        self.submitted.append((task_id, params, instance_type))
        return "job-1"

    def poll(self, job_id):
        self.polls += 1
        return dict(self.status)

    def progress(self, task_id):
        return self.progress_report

    def result_summary(self, task_id):
        return self.summary

    def cancel(self, job_id, reason):
        self.cancelled.append((job_id, reason))

    def tail_logs(self, stream, limit=100):
        return [f"line {i}" for i in range(limit)]

    def fetch_result(self, task_id, output_dir):
        path = f"{output_dir}/{task_id}_result.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("result.json", "{}")
        return path


@pytest.fixture
def api(monkeypatch, tmp_path):
    runner = FakeRunner(tmp_path)
    monkeypatch.setattr(server, "_runner", runner)
    monkeypatch.setattr(server, "_store", server.JobStore(tmp_path / "jobs.json"))
    return TestClient(server.app), runner


def _submit(client, instance="c5.2xlarge", name="g.csv"):
    return client.post("/api/inversion/submit", data={
        "method": "gravity", "mesh_type": "tensor", "instance_type": instance,
        "note": "test", "params_json": json.dumps({"max_iter": 30, "inversion_mode": "single"}),
    }, files=[("files", (name, b"x,y,v\n0,0,1\n", "text/csv"))])


class TestAPI:
    def test_submit_records_the_job(self, api, tmp_path):
        client, runner = api
        r = _submit(client)
        assert r.status_code == 200 and r.json()["job_id"] == "job-1"
        assert runner.submitted[0][2] == "c5.2xlarge"
        saved = json.loads((tmp_path / "jobs.json").read_text())
        assert saved["job-1"]["instance_type"] == "c5.2xlarge"
        assert saved["job-1"]["max_iter"] == 30

    def test_submit_rejects_unknown_instance_and_strips_paths(self, api):
        client, runner = api
        assert _submit(client, instance="x9.huge").status_code == 400
        assert not runner.submitted
        _submit(client, name="../../evil.csv")
        assert runner.uploaded == ["evil.csv"]

    def test_job_list_follows_the_lifecycle(self, api):
        client, runner = api
        _submit(client)
        job = client.get("/api/jobs").json()["jobs"][0]
        assert job["display_status"] == "RUNNABLE" and job["progress"] is None

        runner.status.update(status="RUNNING", started=2000, log_stream="worker/j/1")
        runner.progress_report = {"stage": "inverting", "iteration": 4, "max_iter": 30}
        job = client.get("/api/inversion/job-1").json()
        assert job["display_status"] == "RUNNING" and job["progress"]["iteration"] == 4

        runner.status.update(status="SUCCEEDED", stopped=9000)
        runner.summary = {"n_iterations": 12, "iterations": [{"phi_d": 51.0}],
                          "notes": ["n"], "recovered_model": [1, 2]}
        job = client.get("/api/jobs").json()["jobs"][0]
        assert job["display_status"] == "SUCCEEDED"
        assert job["summary"] == {"n_iterations": 12, "notes": ["n"], "final_phi_d": 51.0}
        polls = runner.polls
        client.get("/api/jobs")
        assert runner.polls == polls   # finished jobs are not queried again

    def test_cancel_shows_cancelled(self, api):
        client, runner = api
        _submit(client)
        runner.status.update(status="RUNNING")
        client.delete("/api/inversion/job-1")
        assert runner.cancelled == [("job-1", server.CANCEL_REASON)]
        runner.status.update(status="FAILED", reason=server.CANCEL_REASON)
        assert client.get("/api/inversion/job-1").json()["display_status"] == "CANCELLED"

    def test_failed_job_keeps_its_error(self, api):
        client, runner = api
        _submit(client)
        runner.status.update(status="FAILED", reason="Essential container exited")
        runner.summary = {"error": "No valid gravity observations"}
        job = client.get("/api/jobs").json()["jobs"][0]
        assert job["display_status"] == "FAILED"
        assert job["summary"]["error"] == "No valid gravity observations"

    def test_logs_and_result(self, api):
        client, runner = api
        _submit(client)
        assert client.get("/api/inversion/job-1/logs").json()["lines"] == []
        runner.status.update(status="RUNNING", log_stream="worker/j/1")
        assert len(client.get("/api/inversion/job-1/logs?limit=5").json()["lines"]) == 5
        r = client.get("/api/inversion/job-1/result")
        assert r.status_code == 200 and r.content[:2] == b"PK"

    def test_jobs_survive_a_restart(self, api, tmp_path, monkeypatch):
        client, runner = api
        _submit(client)
        monkeypatch.setattr(server, "_store", server.JobStore(tmp_path / "jobs.json"))
        assert [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]] == ["job-1"]
