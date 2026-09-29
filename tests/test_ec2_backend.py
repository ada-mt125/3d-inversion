"""EC2 backend: job lifecycle with a fake EC2 API and a fake SSH connection.

The fake EC2 enforces the account's rule that every resource is created with
an ``Owner=<IAM user>`` tag, so a missing tag fails the test.  Nothing here
talks to AWS or opens a network connection.
"""

import io
import json
import tarfile
import zipfile

import boto3
import pytest

from geoinv3d.cloud import ec2 as ec2mod
from geoinv3d.cloud.ec2 import EC2Backend, REMOTE, bootstrap_script, code_archive, start_command
from geoinv3d.cloud.worker import run_local_job
from tests.test_data_pipeline import SMALL_MESH, _station_grid, _synthetic, _write_csv

OWNER = "miaozhou"


@pytest.fixture(autouse=True)
def no_real_aws(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to create a real boto3 client")
    monkeypatch.setattr(boto3, "client", refuse)


class Denied(Exception):
    pass


def _require_owner(specs, types):
    got = {s["ResourceType"]: {t["Key"]: t["Value"] for t in s["Tags"]} for s in specs or []}
    for t in types:
        if got.get(t, {}).get("Owner") != OWNER:
            raise Denied(f"UnauthorizedOperation: {t} created without Owner={OWNER}")


class FakeEC2:
    def __init__(self):
        self.key_pairs, self.groups, self.instances, self.calls = [], {}, {}, []

    def describe_key_pairs(self, Filters):
        name = Filters[0]["Values"][0]
        return {"KeyPairs": [k for k in self.key_pairs if k == name]}

    def create_key_pair(self, KeyName, KeyType, KeyFormat, TagSpecifications):
        _require_owner(TagSpecifications, ["key-pair"])
        self.key_pairs.append(KeyName)
        return {"KeyMaterial": "-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n"}

    def describe_vpcs(self, Filters):
        return {"Vpcs": [{"VpcId": "vpc-1"}]}

    def describe_security_groups(self, Filters):
        name = Filters[0]["Values"][0]
        return {"SecurityGroups": [g for g in self.groups.values() if g["GroupName"] == name]}

    def create_security_group(self, GroupName, VpcId, Description, TagSpecifications):
        _require_owner(TagSpecifications, ["security-group"])
        self.groups["sg-1"] = {"GroupId": "sg-1", "GroupName": GroupName, "IpPermissions": []}
        return {"GroupId": "sg-1"}

    def authorize_security_group_ingress(self, GroupId, IpPermissions, TagSpecifications):
        _require_owner(TagSpecifications, ["security-group-rule"])
        self.groups[GroupId]["IpPermissions"] += IpPermissions
        self.calls.append(("authorize", IpPermissions))

    def describe_images(self, Owners, Filters):
        return {"Images": [{"ImageId": "ami-old", "CreationDate": "2026-01-01"},
                           {"ImageId": "ami-new", "CreationDate": "2026-09-01",
                            "RootDeviceName": "/dev/xvda"}]}

    def run_instances(self, **kw):
        _require_owner(kw["TagSpecifications"], ["instance", "volume", "network-interface"])
        self.calls.append(("run_instances", kw))
        iid = f"i-{len(self.instances) + 1}"
        self.instances[iid] = {"InstanceId": iid, "State": {"Name": "pending"}}
        return {"Instances": [{"InstanceId": iid}]}

    def describe_instances(self, InstanceIds):
        inst = self.instances.get(InstanceIds[0])
        if inst is None:
            raise Exception("InvalidInstanceID.NotFound")
        return {"Reservations": [{"Instances": [inst]}]}

    def set_state(self, iid, state):
        self.instances[iid]["State"]["Name"] = state
        if state == "running":
            self.instances[iid]["PublicIpAddress"] = "203.0.113.7"

    def terminate_instances(self, InstanceIds):
        self.calls.append(("terminate", InstanceIds))
        self.set_state(InstanceIds[0], "terminated")

    def start_instances(self, InstanceIds):
        self.calls.append(("start", InstanceIds))
        self.set_state(InstanceIds[0], "running")


class FakeSSH:
    """In-memory remote file system; commands are recorded, not run."""

    def __init__(self, fs, commands):
        self.fs, self.commands = fs, commands

    def read(self, path):
        return self.fs.get(path)

    def write(self, path, data):
        self.fs[path] = data

    def put(self, local, remote):
        self.fs[remote] = open(local, "rb").read()

    def get(self, remote, local):
        open(local, "wb").write(self.fs[remote])

    def run(self, cmd, timeout=60):
        self.commands.append(cmd)
        if "geoinv3d.cloud.worker --local" in cmd:
            self.fs[f"{REMOTE}/job/STARTED"] = b"1790000000\n"
            return 0, "started\n"
        if "tail" in cmd:
            return 0, "line 1\nline 2\n"
        return 0, ""

    def close(self):
        pass


@pytest.fixture
def backend(tmp_path):
    ec2, fs, commands = FakeEC2(), {}, []
    b = EC2Backend("ap-south-1", owner=OWNER, ec2=ec2, state_dir=tmp_path / "state",
                   ssh_factory=lambda host, key: FakeSSH(fs, commands),
                   ip_lookup=lambda: "198.51.100.9")
    b.fake_ec2, b.fs, b.commands = ec2, fs, commands
    return b


def _start(backend, tmp_path):
    data = tmp_path / "g.csv"
    data.write_bytes(b"x,y,v\n0,0,1\n")
    job_id = backend.start_job("t1", [str(data)], {"max_iter": 30}, "c5.2xlarge")
    return job_id, {"job_id": job_id, "task_id": "t1", **backend.initial_fields()}


def _refresh(backend, record):
    fields = backend.refresh(record)
    thread = backend._threads.get(record["job_id"])
    if thread is not None:
        thread.join(timeout=10)
    record.update(fields)
    record.update(backend.refresh(record))   # pick up what the thread did
    return record


class TestLaunch:
    def test_instance_is_tagged_guarded_and_reachable_only_from_here(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        run = dict(backend.fake_ec2.calls)["run_instances"]
        assert run["ImageId"] == "ami-new" and run["InstanceType"] == "c5.2xlarge"
        assert run["InstanceInitiatedShutdownBehavior"] == "stop"
        assert run["MetadataOptions"]["HttpTokens"] == "required"
        assert run["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
        tags = {t["Key"]: t["Value"] for t in run["TagSpecifications"][0]["Tags"]}
        assert tags == {"Owner": OWNER, "Project": "geoinv3d", "Name": "geoinv3d-t1",
                        "geoinv3d:task": "t1"}
        assert "shutdown -h +60" in run["UserData"]
        rule = dict(backend.fake_ec2.calls)["authorize"][0]
        assert rule["FromPort"] == 22 and rule["IpRanges"][0]["CidrIp"] == "198.51.100.9/32"
        assert backend.key_path.exists()
        staged = backend.state_dir / "staging" / "t1"
        assert sorted(p.name for p in staged.iterdir()) == ["g.csv", "params.json"]
        assert record["status"] == "STARTING" and record["phase"] == "launching"

    def test_resources_are_reused(self, backend, tmp_path):
        _start(backend, tmp_path)
        _start(backend, tmp_path)
        assert len(backend.fake_ec2.key_pairs) == 1 and len(backend.fake_ec2.groups) == 1
        assert sum(1 for c in backend.fake_ec2.calls if c[0] == "authorize") == 1

    def test_key_in_aws_without_local_private_key_gets_a_new_name(self, backend, tmp_path):
        backend.fake_ec2.key_pairs.append(f"geoinv3d-{OWNER}")
        backend.ensure_key_pair()
        assert backend.key_name.startswith(f"geoinv3d-{OWNER}-")

    def test_missing_owner_tag_would_be_denied(self, backend):
        with pytest.raises(Denied):
            backend.fake_ec2.run_instances(TagSpecifications=[
                {"ResourceType": "instance", "Tags": [{"Key": "Project", "Value": "x"}]}])


class TestLifecycle:
    def test_full_run(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        assert _refresh(backend, record)["progress"] == {"stage": "launching"}

        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/READY"] = b""
        _refresh(backend, record)
        assert record["phase"] == "running" and record["status"] == "RUNNING"
        assert backend.fs[f"{REMOTE}/job/data/g.csv"] == b"x,y,v\n0,0,1\n"
        assert json.loads(backend.fs[f"{REMOTE}/job/params.json"]) == {"max_iter": 30}
        assert any("--local params.json data out" in c for c in backend.commands)
        assert not (backend.state_dir / "staging" / "t1").exists()

        backend.fs[f"{REMOTE}/job/out/progress.json"] = b'{"stage": "inverting", "iteration": 3}'
        assert _refresh(backend, record)["progress"]["iteration"] == 3

        backend.fs[f"{REMOTE}/job/out/exit_code"] = b"0\n"
        backend.fs[f"{REMOTE}/job/out/result.json"] = b'{"n_iterations": 9}'
        backend.fs[f"{REMOTE}/job/out/result.zip"] = b"PK..."
        backend.fs[f"{REMOTE}/job/out/worker.log"] = b"a\nb\nc\n"
        _refresh(backend, record)
        assert record["phase"] == "succeeded" and record["status"] == "SUCCEEDED"
        assert ("terminate", [job_id]) in backend.fake_ec2.calls
        assert backend.result_summary(record) == {"n_iterations": 9}
        assert backend.fetch_result(record).endswith("result.zip")
        assert backend.tail_logs(record, limit=2) == ["b", "c"]

    def test_worker_failure(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        record["phase"] = "running"
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/job/out/exit_code"] = b"1\n"
        backend.fs[f"{REMOTE}/job/out/result.json"] = b'{"error": "No valid gravity data"}'
        _refresh(backend, record)
        assert record["status"] == "FAILED" and record["reason"] == "No valid gravity data"
        assert ("terminate", [job_id]) in backend.fake_ec2.calls

    def test_finishing_shows_that_the_result_is_being_collected(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        record["phase"] = "running"
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/job/out/progress.json"] = b'{"stage": "done", "iteration": 30}'
        backend.fs[f"{REMOTE}/job/out/exit_code"] = b"0\n"
        fields = backend.refresh(record)
        assert fields["phase"] == "finishing" and fields["progress"]["stage"] == "collecting"
        backend._threads[job_id].join(timeout=10)

    def test_outcome_is_recorded_before_the_instance_is_terminated(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        record["phase"] = "running"
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/job/out/exit_code"] = b"0\n"
        backend.fs[f"{REMOTE}/job/out/result.json"] = b'{"n_iterations": 3}'
        seen, terminate = [], backend.fake_ec2.terminate_instances

        def spy(**kw):
            seen.append(backend._live.get(job_id, {}).get("phase"))
            return terminate(**kw)

        backend.fake_ec2.terminate_instances = spy
        _refresh(backend, record)
        assert seen == ["succeeded"] and record["phase"] == "succeeded"

    def test_bootstrap_failure_terminates(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/BOOTSTRAP_FAILED"] = b""
        _refresh(backend, record)
        assert record["phase"] == "failed" and "Installing the software failed" in record["reason"]
        assert ("terminate", [job_id]) in backend.fake_ec2.calls

    def test_stopped_before_start_is_a_failure(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        backend.fake_ec2.set_state(job_id, "stopped")
        assert _refresh(backend, record)["status"] == "FAILED"

    def test_stopped_after_finishing_can_be_fetched(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        record["phase"] = "running"
        backend.fake_ec2.set_state(job_id, "stopped")
        backend.fake_ec2.instances[job_id]["StateTransitionReason"] =             "User initiated (2026-09-27 12:00:00 GMT)"
        assert _refresh(backend, record)["status"] == "STOPPED"
        assert record["stopped"] == 1790510400000   # billing ended when it stopped
        backend.fs[f"{REMOTE}/job/out/exit_code"] = b"0\n"
        backend.fs[f"{REMOTE}/job/out/result.json"] = b'{"n_iterations": 4}'
        backend.fetch(record)
        backend._threads[job_id].join(timeout=10)
        record.update(backend.refresh(record))
        assert ("start", [job_id]) in backend.fake_ec2.calls
        assert record["status"] == "SUCCEEDED"

    def test_cancel_terminates(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        backend.cancel(record)
        record.update(backend.refresh(record))
        assert ("terminate", [job_id]) in backend.fake_ec2.calls
        assert record["phase"] == "cancelled" and record["status"] == "FAILED"

    def test_stop_and_keep_writes_the_flag_and_leaves_the_instance(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        with pytest.raises(RuntimeError, match="not started"):
            backend.request_finish(record)
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/READY"] = b""
        _refresh(backend, record)
        assert record["phase"] == "running"
        backend.request_finish(record)
        assert f"{REMOTE}/job/out/STOP" in backend.fs
        # the worker then finishes as usual: nothing is terminated here
        assert ("terminate", [job_id]) not in backend.fake_ec2.calls
        assert _refresh(backend, record)["status"] == "RUNNING"

    def test_instance_terminated_elsewhere(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        backend.fake_ec2.set_state(job_id, "terminated")
        assert _refresh(backend, record)["status"] == "FAILED"


class TestScripts:
    def test_bootstrap_installs_pinned_packages_and_marks_ready(self):
        script = bootstrap_script(45)
        assert script.startswith("#!/bin/bash")
        assert "shutdown -h +45" in script and "simpeg==0.25.2" in script
        assert "touch /opt/geoinv3d/READY" in script and "BOOTSTRAP_FAILED" in script

    def test_start_command_guards_runtime_and_leaves_time_to_fetch(self):
        cmd = start_command("t1", 720, 30)
        assert "shutdown -h +720" in cmd and "shutdown -h +30" in cmd
        assert "echo $? > out/exit_code" in cmd and "nohup setsid" in cmd
        # only the worker goes to the background, so the command returns at once
        # (a bare "a && b & echo started" backgrounds the whole list and holds
        # the SSH channel open until the worker ends)
        assert "&& { nohup setsid" in cmd and "& } && echo started" in cmd
        assert "date +%s > STARTED" in cmd

    def test_retried_setup_does_not_start_the_worker_twice(self, backend, tmp_path):
        job_id, record = _start(backend, tmp_path)
        backend.fake_ec2.set_state(job_id, "running")
        backend.fs[f"{REMOTE}/READY"] = b""
        backend.fs[f"{REMOTE}/job/STARTED"] = b"1790000000\n"   # an earlier attempt
        _refresh(backend, record)
        assert record["phase"] == "running" and record["started"] == 1790000000 * 1000
        assert not any("--local" in c for c in backend.commands)
        assert f"{REMOTE}/code.tar.gz" not in backend.fs

    def test_code_archive_has_the_worker_and_no_bytecode(self):
        with tarfile.open(fileobj=io.BytesIO(code_archive())) as tar:
            names = tar.getnames()
        assert "geoinv3d/cloud/worker.py" in names
        assert not any("__pycache__" in n for n in names)
        with tarfile.open(fileobj=io.BytesIO(code_archive())) as tar:
            # writable by the owner, so a second upload can replace the code
            assert all(m.mode == (0o755 if m.isdir() else 0o644) for m in tar.getmembers())


class TestLocalWorker:
    def test_local_job_writes_progress_and_results(self, tmp_path):
        locs = _station_grid(0.0)
        (tmp_path / "data").mkdir()
        _write_csv(tmp_path / "data" / "g.csv", locs, _synthetic("gravity", locs))
        params = {"method_type": "gravity", "inversion_mode": "single",
                  "datasets": [{"method": "gravity", "files": ["g.csv"], "noise_pct": 0.05,
                                "noise_floor": 0.01}],
                  "param_mode": "manual", "regularization_type": "l2", "max_iter": 6,
                  "topography": {"flat_elevation": 0}, "mesh_type": "tensor", **SMALL_MESH}
        (tmp_path / "params.json").write_text(json.dumps(params))
        code = run_local_job(str(tmp_path / "params.json"), str(tmp_path / "data"),
                             str(tmp_path / "out"))
        assert code == 0
        out = tmp_path / "out"
        assert json.loads((out / "progress.json").read_text())["stage"] == "done"
        assert json.loads((out / "result.json").read_text())["n_iterations"] > 0
        with zipfile.ZipFile(out / "result.zip") as zf:
            assert "recovered_model.npy" in zf.namelist()

    def test_stop_flag_ends_the_inversion_and_keeps_the_result(self, tmp_path):
        from geoinv3d.methods.directives import IterationCollector
        locs = _station_grid(0.0)
        (tmp_path / "data").mkdir()
        _write_csv(tmp_path / "data" / "g.csv", locs, _synthetic("gravity", locs))
        params = {"method_type": "gravity", "inversion_mode": "single",
                  "datasets": [{"method": "gravity", "files": ["g.csv"], "noise_pct": 0.05,
                                "noise_floor": 0.01}],
                  "param_mode": "manual", "regularization_type": "l2", "max_iter": 6,
                  "topography": {"flat_elevation": 0}, "mesh_type": "tensor", **SMALL_MESH}
        (tmp_path / "params.json").write_text(json.dumps(params))
        out = tmp_path / "out"
        out.mkdir()
        (out / "STOP").write_text("stop requested\n")   # as EC2Backend.request_finish writes it
        assert run_local_job(str(tmp_path / "params.json"), str(tmp_path / "data"), str(out)) == 0
        meta = json.loads((out / "result.json").read_text())
        assert meta["stopped_early"] == {"reason": "stopped by the user", "at_iteration": 1}
        assert meta["converged"] is False and meta["n_iterations"] == 1
        with zipfile.ZipFile(out / "result.zip") as zf:
            assert "recovered_model.npy" in zf.namelist()
        progress = json.loads((out / "progress.json").read_text())
        assert progress["stage"] == "done" and "stopped early" in progress["message"]
        assert IterationCollector.stop_check is None   # reset for whatever runs next

    def test_local_job_failure(self, tmp_path):
        (tmp_path / "params.json").write_text(json.dumps(
            {"method_type": "gravity", "inversion_mode": "single",
             "datasets": [{"method": "gravity", "files": ["missing.csv"]}]}))
        (tmp_path / "data").mkdir()
        assert run_local_job(str(tmp_path / "params.json"), str(tmp_path / "data"),
                             str(tmp_path / "out")) == 1
        assert "missing.csv" in json.loads((tmp_path / "out" / "result.json").read_text())["error"]
