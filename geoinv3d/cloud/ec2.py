"""EC2 backend: one tagged EC2 instance per job, driven over SSH.

For an AWS account where the user may only use EC2 and EBS, and every
resource must carry ``Owner=<IAM user name>`` when it is created.  No S3,
ECR, Batch or IAM is needed:

1. ``start_job`` stages the files locally and launches an Amazon Linux 2023
   instance (tagged Owner, Project=geoinv3d, Name) whose user data installs
   Python and the pinned packages with uv, then touches ``READY``.
2. A background thread waits for SSH and READY, uploads the geoinv3d code,
   the data and params.json, and starts
   ``python -m geoinv3d.cloud.worker --local`` in the background.
3. ``refresh`` reads the worker's progress.json over SSH; when the worker
   has written ``exit_code``, a thread downloads result.zip, result.json and
   the log to ``~/.geoinv3d/results/<task_id>/`` and terminates the instance.

Safety: the instance shuts itself down (``InstanceInitiatedShutdownBehavior
= stop``, so only its disk is billed) if no job starts within
``bootstrap_timeout_min``, if the job runs longer than ``run_timeout_min``,
and ``grace_min`` after the job ends if nobody fetched the result (the API
server was not running).  ``fetch`` restarts such an instance to collect the
result.  SSH is open only to this machine's public IP.

Job records (kept by the API server) carry a ``phase``: launching,
bootstrapping, uploading, running, finishing, succeeded, failed, cancelled,
or stopped (instance stopped before the result was fetched).
"""

from __future__ import annotations

import calendar
import io
import json
import re
import shutil
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ..io.filestore import clone_or_copy

REMOTE = "/opt/geoinv3d"
PYTHON_VERSION = "3.13"
# The versions tested locally; the instance installs exactly these.
PINNED_PACKAGES = ("numpy==2.3.3 scipy==1.16.2 simpeg==0.25.2 discretize==0.12.0 "
                   "numba==0.67.0 rasterio==1.5.0 scikit-learn==1.8.0")   # scikit-learn: PGI
PHASE_STATUS = {
    "launching": "STARTING", "bootstrapping": "STARTING", "uploading": "STARTING",
    "running": "RUNNING", "finishing": "RUNNING",
    "succeeded": "SUCCEEDED", "failed": "FAILED", "cancelled": "FAILED",
    "stopped": "STOPPED",
}
TERMINAL_PHASES = ("succeeded", "failed", "cancelled")
# Progress stage shown while the instance is being prepared
PROVISION_STAGE = {"launching": "launching", "bootstrapping": "bootstrapping",
                   "uploading": "sending_data"}


def bootstrap_script(bootstrap_timeout_min: int) -> str:
    """User data: install Python and the packages, then mark READY."""
    return f"""#!/bin/bash
# GeoInv3D worker bootstrap (Amazon Linux 2023)
mkdir -p {REMOTE}
exec > /var/log/geoinv3d-bootstrap.log 2>&1
set -euxo pipefail
trap 'touch {REMOTE}/BOOTSTRAP_FAILED' ERR
# Stop (not terminate) the instance if no job has started by then
shutdown -h +{bootstrap_timeout_min}
dnf install -y python3-pip tar
python3 -m pip install --quiet "uv>=0.5"
export UV_PYTHON_INSTALL_DIR={REMOTE}/python
uv venv --python {PYTHON_VERSION} {REMOTE}/venv
uv pip install --python {REMOTE}/venv/bin/python {PINNED_PACKAGES}
mkdir -p {REMOTE}/job/data {REMOTE}/job/out {REMOTE}/code
chown -R ec2-user:ec2-user {REMOTE}
touch {REMOTE}/READY
"""


def start_command(task_id: str, run_timeout_min: int, grace_min: int) -> str:
    """Start the worker detached; afterwards leave ``grace_min`` to fetch the result."""
    inner = (f"PYTHONPATH={REMOTE}/code TASK_ID={task_id} {REMOTE}/venv/bin/python "
             f"-m geoinv3d.cloud.worker --local params.json data out > out/worker.log 2>&1; "
             f"echo $? > out/exit_code; sudo shutdown -c; sudo shutdown -h +{grace_min}")
    # The braces matter: in "a && b & echo" bash backgrounds the whole a && b
    # list, which keeps the SSH channel open until the worker ends.  STARTED
    # marks the job as started so that a retried set-up never starts it twice.
    return (f"cd {REMOTE}/job && (sudo shutdown -c || true) && sudo shutdown -h +{run_timeout_min} "
            f"&& date +%s > STARTED "
            f"&& {{ nohup setsid bash -c '{inner}' > /dev/null 2>&1 < /dev/null & }} && echo started")


def _archive_entry(ti: tarfile.TarInfo):
    if "__pycache__" in ti.name:
        return None
    # Fixed modes: on Windows (e.g. OneDrive folders marked read-only) the
    # directories come out r-x, and a later upload could not replace the code
    ti.mode = 0o755 if ti.isdir() else 0o644
    return ti


def code_archive() -> bytes:
    """tar.gz of the installed geoinv3d package (the worker's code)."""
    import geoinv3d
    pkg = Path(geoinv3d.__file__).parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(pkg, arcname="geoinv3d", filter=_archive_entry)
    return buf.getvalue()


class SSH:
    """Minimal paramiko wrapper: run commands, read and write files."""

    def __init__(self, host: str, key_path: str, user: str = "ec2-user",
                 timeout: float = 10.0) -> None:
        import paramiko
        self.client = paramiko.SSHClient()
        # Each job gets a fresh instance, so its host key is new by design
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(host, username=user, key_filename=key_path, timeout=timeout,
                            banner_timeout=timeout, auth_timeout=timeout,
                            look_for_keys=False, allow_agent=False)
        self.sftp = self.client.open_sftp()

    def run(self, cmd: str, timeout: float = 60.0) -> tuple[int, str]:
        _, out, err = self.client.exec_command(cmd, timeout=timeout)
        code = out.channel.recv_exit_status()
        return code, out.read().decode(errors="replace") + err.read().decode(errors="replace")

    def read(self, path: str) -> Optional[bytes]:
        try:
            with self.sftp.open(path, "rb") as f:
                # Request all blocks at once: reading them one round trip at a
                # time managed ~30 KB/s between the UK and ap-south-1
                f.prefetch()
                return f.read()
        except OSError:
            return None

    def write(self, path: str, data: bytes) -> None:
        with self.sftp.open(path, "wb") as f:
            f.set_pipelined(True)
            f.write(data)

    def put(self, local: str, remote: str) -> None:
        self.sftp.put(local, remote)

    def get(self, remote: str, local: str) -> None:
        self.sftp.get(remote, local)

    def close(self) -> None:
        self.client.close()


def transition_time_ms(inst: dict) -> Optional[int]:
    """When the instance last changed state, from e.g.
    ``StateTransitionReason = "User initiated (2026-09-27 12:00:00 GMT)"``."""
    m = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)",
                  inst.get("StateTransitionReason", ""))
    if not m:
        return None
    return calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) * 1000


def my_public_ip() -> str:
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as r:
        return r.read().decode().strip()


class EC2Backend:
    """Job backend for the API server (see the module docstring)."""

    def __init__(self, region: str, owner: Optional[str] = None, ec2=None,
                 state_dir=None, ssh_factory: Optional[Callable] = None,
                 ip_lookup: Callable[[], str] = my_public_ip, clock=time.time,
                 root_volume_gb: int = 30, bootstrap_timeout_min: int = 60,
                 run_timeout_min: int = 12 * 60, grace_min: int = 30) -> None:
        import boto3
        self.region = region
        self.ec2 = ec2 or boto3.client("ec2", region_name=region)
        self.owner = owner or boto3.client("sts").get_caller_identity()["Arn"].split("/")[-1]
        self.state_dir = Path(state_dir or Path.home() / ".geoinv3d")
        self.ssh_factory = ssh_factory or SSH
        self.ip_lookup, self.clock = ip_lookup, clock
        self.root_volume_gb = root_volume_gb
        self.bootstrap_timeout_min = bootstrap_timeout_min
        self.run_timeout_min, self.grace_min = run_timeout_min, grace_min
        self.key_name = f"geoinv3d-{self.owner}"
        self.sg_name = f"geoinv3d-ssh-{self.owner}"
        self._live: dict[str, dict] = {}      # job id -> fields set by background threads
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._setup_lock = threading.Lock()

    # ── tags ──
    def tags(self, **extra) -> list[dict]:
        tags = {"Owner": self.owner, "Project": "geoinv3d", **extra}
        return [{"Key": k, "Value": v} for k, v in tags.items()]

    # ── account resources (created once, tagged) ──
    @property
    def key_path(self) -> Path:
        return self.state_dir / "keys" / f"{self.key_name}.pem"

    def ensure_key_pair(self) -> str:
        """The SSH key pair: created in AWS with its private key kept locally."""
        if self.key_path.exists():
            return self.key_name
        existing = self.ec2.describe_key_pairs(
            Filters=[{"Name": "key-name", "Values": [self.key_name]}])["KeyPairs"]
        if existing:   # key in AWS but its private half is not on this machine
            self.key_name = f"geoinv3d-{self.owner}-{int(self.clock())}"
        resp = self.ec2.create_key_pair(
            KeyName=self.key_name, KeyType="ed25519", KeyFormat="pem",
            TagSpecifications=[{"ResourceType": "key-pair", "Tags": self.tags()}])
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_text(resp["KeyMaterial"], encoding="utf-8")
        self.key_path.chmod(0o600)
        return self.key_name

    def default_vpc(self) -> str:
        vpcs = self.ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
        if not vpcs:
            raise RuntimeError(f"No default VPC in {self.region}")
        return vpcs[0]["VpcId"]

    def ensure_security_group(self) -> str:
        """Security group allowing SSH from this machine's current public IP only."""
        vpc = self.default_vpc()
        groups = self.ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [self.sg_name]},
            {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
        if groups:
            sg = groups[0]
        else:
            gid = self.ec2.create_security_group(
                GroupName=self.sg_name, VpcId=vpc,
                Description="GeoInv3D workers: SSH from the submitting machine only",
                TagSpecifications=[{"ResourceType": "security-group",
                                    "Tags": self.tags()}])["GroupId"]
            sg = {"GroupId": gid, "IpPermissions": []}
        cidr = f"{self.ip_lookup()}/32"
        allowed = {r["CidrIp"] for p in sg.get("IpPermissions", []) if p.get("FromPort") == 22
                   for r in p.get("IpRanges", [])}
        if cidr not in allowed:
            try:
                self.ec2.authorize_security_group_ingress(
                    GroupId=sg["GroupId"],
                    IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                                    "IpRanges": [{"CidrIp": cidr,
                                                  "Description": "GeoInv3D submitting machine"}]}],
                    TagSpecifications=[{"ResourceType": "security-group-rule",
                                        "Tags": self.tags()}])
            except Exception as e:
                # another job (or process) added the same rule first
                if "InvalidPermission.Duplicate" not in str(e):
                    raise
        return sg["GroupId"]

    def latest_ami(self) -> dict:
        images = self.ec2.describe_images(Owners=["amazon"], Filters=[
            {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
            {"Name": "state", "Values": ["available"]}])["Images"]
        if not images:
            raise RuntimeError("No Amazon Linux 2023 image found")
        return max(images, key=lambda i: i["CreationDate"])

    # ── backend interface ──
    def start_job(self, task_id: str, files: list[str], params: dict,
                  instance_type: str) -> str:
        """Stage the files, launch the instance; returns the instance id as job id."""
        staging = self.state_dir / "staging" / task_id
        staging.mkdir(parents=True, exist_ok=True)
        for f in files:
            clone_or_copy(f, staging / Path(f).name)
        (staging / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

        with self._setup_lock:   # parallel jobs would create the key or the rule twice
            key = self.ensure_key_pair()
            sg = self.ensure_security_group()
        ami = self.latest_ami()
        name = f"geoinv3d-{task_id}"
        resp = self.ec2.run_instances(
            ImageId=ami["ImageId"], InstanceType=instance_type, MinCount=1, MaxCount=1,
            KeyName=key, SecurityGroupIds=[sg],
            InstanceInitiatedShutdownBehavior="stop",
            UserData=bootstrap_script(self.bootstrap_timeout_min),
            BlockDeviceMappings=[{"DeviceName": ami.get("RootDeviceName", "/dev/xvda"),
                                  "Ebs": {"VolumeSize": self.root_volume_gb,
                                          "VolumeType": "gp3",
                                          "DeleteOnTermination": True}}],
            MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": self.tags(Name=name, **{"geoinv3d:task": task_id})},
                {"ResourceType": "volume", "Tags": self.tags(Name=name, **{"geoinv3d:task": task_id})},
                {"ResourceType": "network-interface",
                 "Tags": self.tags(Name=name, **{"geoinv3d:task": task_id})},
            ])
        return resp["Instances"][0]["InstanceId"]

    def initial_fields(self) -> dict:
        return {"phase": "launching", "status": "STARTING", "backend": "ec2",
                "key_name": self.key_name, "created": int(self.clock() * 1000)}

    def _instance(self, instance_id: str) -> Optional[dict]:
        try:
            res = self.ec2.describe_instances(InstanceIds=[instance_id])["Reservations"]
        except Exception as e:
            if "InvalidInstanceID" in str(e):
                return None
            raise
        return res[0]["Instances"][0] if res else None

    def _ssh(self, inst: dict, record: dict):
        key = self.state_dir / "keys" / f"{record.get('key_name', self.key_name)}.pem"
        return self.ssh_factory(inst["PublicIpAddress"], str(key))

    def refresh(self, record: dict) -> dict:
        """New fields for a job record: phase, status, progress, times, summary."""
        job_id, phase = record["job_id"], record.get("phase", "launching")
        with self._lock:
            live = dict(self._live.get(job_id, {}))
        if live.get("phase"):
            phase = live["phase"]
        fields = dict(live)
        if phase in TERMINAL_PHASES:
            fields.setdefault("phase", phase)
            return self._with_status(fields)

        inst = self._instance(job_id)
        if inst is None and phase == "launching" \
                and self.clock() * 1000 - record.get("created", 0) < 120_000:
            # describe_instances may not know an instance launched seconds ago (with several
            # launches at once two of eight were reported gone and left running untracked)
            fields["progress"] = {"stage": "launching"}
            return self._with_status(fields)
        state = inst["State"]["Name"] if inst else "terminated"
        fields["instance_state"] = state
        if state in ("terminated", "shutting-down") and phase not in TERMINAL_PHASES:
            fields.update(phase="failed", reason="The instance was terminated outside GeoInv3D")
        elif state in ("stopped", "stopping"):
            # Billing of the instance ended when it stopped
            fields["stopped"] = record.get("stopped") or transition_time_ms(inst)                 or int(self.clock() * 1000)
            if phase in ("launching", "bootstrapping", "uploading"):
                fields.update(phase="failed",
                              reason="The instance stopped before the job started "
                                     f"(set-up took more than {self.bootstrap_timeout_min} min)")
            elif phase != "finishing":
                fields.update(phase="stopped",
                              reason="The instance stopped before the result was fetched")
        elif state == "running" and phase in ("launching", "bootstrapping", "uploading"):
            fields["phase"] = phase if phase != "launching" else "bootstrapping"
            self._spawn(job_id, record, self._provision)
            fields["progress"] = {"stage": PROVISION_STAGE[fields["phase"]]}
        elif state == "running" and phase == "running":
            try:
                ssh = self._ssh(inst, record)
                try:
                    report = ssh.read(f"{REMOTE}/job/out/progress.json")
                    if report:
                        fields["progress"] = json.loads(report)
                    if ssh.read(f"{REMOTE}/job/out/exit_code") is not None:
                        fields["phase"] = "finishing"
                        fields["progress"] = {**fields.get("progress", {}), "stage": "collecting"}
                        self._spawn(job_id, record, self._finish)
                finally:
                    ssh.close()
            except Exception as e:
                fields["refresh_error"] = f"SSH: {e}"
        elif state == "running" and phase == "finishing":
            self._spawn(job_id, record, self._finish)   # e.g. after an API restart
        elif phase == "launching":
            fields["progress"] = {"stage": "launching"}
        return self._with_status(fields)

    @staticmethod
    def _with_status(fields: dict) -> dict:
        fields["status"] = PHASE_STATUS.get(fields.get("phase"), "STARTING")
        return fields

    def _set(self, job_id: str, **fields) -> None:
        with self._lock:
            self._live.setdefault(job_id, {}).update(fields)

    def _spawn(self, job_id: str, record: dict, target) -> None:
        with self._lock:
            t = self._threads.get(job_id)
            if t is not None and t.is_alive():
                return
            t = threading.Thread(target=self._guard, args=(job_id, dict(record), target),
                                 daemon=True)
            self._threads[job_id] = t
        t.start()

    def _guard(self, job_id, record, target) -> None:
        try:
            target(job_id, record)
        except Exception as e:
            self._set(job_id, refresh_error=str(e))

    def _wait_ssh(self, job_id, record, timeout_s=600, wait=10):
        t0 = self.clock()
        while True:
            inst = self._instance(job_id)
            if inst and inst["State"]["Name"] == "running" and inst.get("PublicIpAddress"):
                try:
                    return self._ssh(inst, record)
                except Exception:
                    pass
            if self.clock() - t0 > timeout_s:
                raise TimeoutError("SSH to the instance did not come up")
            time.sleep(wait)

    def _provision(self, job_id: str, record: dict) -> None:
        """Wait for set-up, upload code and data, start the worker."""
        self._set(job_id, phase="bootstrapping")
        ssh = self._wait_ssh(job_id, record)
        try:
            while True:
                if ssh.read(f"{REMOTE}/BOOTSTRAP_FAILED") is not None:
                    _, log = ssh.run("sudo tail -n 20 /var/log/geoinv3d-bootstrap.log")
                    self._set(job_id, phase="failed",
                              reason="Installing the software failed:\n" + log)
                    self.ec2.terminate_instances(InstanceIds=[job_id])
                    return
                if ssh.read(f"{REMOTE}/READY") is not None:
                    break
                time.sleep(10)
            task_id = record["task_id"]
            staging = self.state_dir / "staging" / task_id
            marker = ssh.read(f"{REMOTE}/job/STARTED")
            if marker is not None:
                # An earlier attempt started the worker (and then lost the
                # connection): follow that job rather than start it again
                try:
                    started = int(marker.strip()) * 1000
                except ValueError:
                    started = int(self.clock() * 1000)
                self._set(job_id, phase="running", started=started,
                          progress={"stage": "starting"})
                shutil.rmtree(staging, ignore_errors=True)
                return
            self._set(job_id, phase="uploading")
            ssh.write(f"{REMOTE}/code.tar.gz", code_archive())
            code, out = ssh.run(f"chmod -R u+w {REMOTE}/code/geoinv3d 2>/dev/null; "
                                f"rm -rf {REMOTE}/code/geoinv3d && "
                                f"tar xzf {REMOTE}/code.tar.gz -C {REMOTE}/code")
            if code != 0:
                raise RuntimeError(f"Unpacking the code failed: {out}")
            for f in sorted(staging.iterdir()):
                dest = "params.json" if f.name == "params.json" else f"data/{f.name}"
                ssh.put(str(f), f"{REMOTE}/job/{dest}")
            code, out = ssh.run(start_command(task_id, self.run_timeout_min, self.grace_min))
            if code != 0 or "started" not in out:
                raise RuntimeError(f"Starting the worker failed: {out}")
            self._set(job_id, phase="running", started=int(self.clock() * 1000),
                      progress={"stage": "starting"})
            shutil.rmtree(staging, ignore_errors=True)
        finally:
            ssh.close()

    def _finish(self, job_id: str, record: dict) -> None:
        """Download the result and the log, then terminate the instance."""
        self._set(job_id, phase="finishing", progress={"stage": "collecting"})
        ssh = self._wait_ssh(job_id, record)
        out_dir = self.result_dir(record)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            exit_code = (ssh.read(f"{REMOTE}/job/out/exit_code") or b"").strip()
            for name in ("result.zip", "result.json", "worker.log", "progress.json"):
                data = ssh.read(f"{REMOTE}/job/out/{name}")
                if data is not None:
                    (out_dir / name).write_bytes(data)
        finally:
            ssh.close()
        summary = self._summary(out_dir)
        ok = exit_code == b"0"
        fields = dict(phase="succeeded" if ok else "failed", stopped=int(self.clock() * 1000),
                      progress=None)
        if not ok:
            fields["reason"] = (summary or {}).get("error") or \
                f"The worker exited with status {exit_code.decode() or 'unknown'}"
        # Record the outcome first: a refresh that sees the instance shutting
        # down while the phase is still "finishing" reports it as terminated
        # outside GeoInv3D
        self._set(job_id, **fields)
        self.ec2.terminate_instances(InstanceIds=[job_id])

    def result_dir(self, record: dict) -> Path:
        return self.state_dir / "results" / record["task_id"]

    @staticmethod
    def _summary(out_dir: Path) -> Optional[dict]:
        path = out_dir / "result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def result_summary(self, record: dict) -> Optional[dict]:
        return self._summary(self.result_dir(record))

    def cancel(self, record: dict) -> None:
        self.ec2.terminate_instances(InstanceIds=[record["job_id"]])
        self._set(record["job_id"], phase="cancelled", stopped=int(self.clock() * 1000),
                  reason="Cancelled by user")

    def request_finish(self, record: dict) -> None:
        """Ask the worker to stop after its current iteration and keep the result.

        Writes job/out/STOP, which the worker checks after every iteration (and
        every coordinate-descent sweep); it then packs its result and exits as
        usual, and the result is fetched and the instance ends as for any job.
        """
        if record.get("phase") != "running":
            raise RuntimeError("The inversion has not started yet: there is nothing to keep")
        inst = self._instance(record["job_id"])
        if not inst or inst["State"]["Name"] != "running":
            raise RuntimeError("The instance is not running")
        ssh = self._ssh(inst, record)
        try:
            ssh.write(f"{REMOTE}/job/out/STOP", b"stop requested\n")
        finally:
            ssh.close()

    def fetch(self, record: dict) -> None:
        """Start a stopped instance again and collect its result."""
        self.ec2.start_instances(InstanceIds=[record["job_id"]])
        self._set(record["job_id"], phase="finishing", stopped=None)
        self._spawn(record["job_id"], record, self._finish)

    def tail_logs(self, record: dict, limit: int = 100) -> list[str]:
        phase = record.get("phase")
        local = self.result_dir(record) / "worker.log"
        if phase in TERMINAL_PHASES and local.exists():
            return local.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        inst = self._instance(record["job_id"])
        if not inst or inst["State"]["Name"] != "running" or not inst.get("PublicIpAddress"):
            return []
        ssh = self._ssh(inst, record)
        try:
            path = f"{REMOTE}/job/out/worker.log" if phase in ("running", "finishing") \
                else "/var/log/geoinv3d-bootstrap.log"
            _, out = ssh.run(f"sudo tail -n {int(limit)} {path}")
            return out.splitlines()
        finally:
            ssh.close()

    def fetch_result(self, record: dict, output_dir=None) -> str:
        path = self.result_dir(record) / "result.zip"
        if not path.exists():
            raise FileNotFoundError("The result has not been downloaded yet")
        return str(path)
