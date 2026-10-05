"""Run a parameter sweep on a few EC2 instances, each running its share of the jobs.

    python deploy/ec2_sweep.py RUNS.json DATA_DIR OUT_DIR [--instances 8] [--type c5.4xlarge]
                               [--parallel 3] [--only PREFIX,...] [--resume STAMP]

``RUNS.json`` maps a job name to {"params": ..., "files": [...]} (e.g. the ``runs.json`` of
examples/output/synthetic_ablation_lp); the files are read from DATA_DIR.  The jobs are dealt
round-robin to ``--instances`` instances (Amazon Linux 2023, tagged Owner=<IAM user>,
Project=geoinv3d, set up as by the EC2 backend), and each runs deploy/sweep_runner.py on its
share, ``--parallel`` at a time.  Progress is printed every half minute; when an instance is
done its results are downloaded to OUT_DIR/<job>/ (result.zip, result.json, worker.log,
minutes, exit_code) and it is terminated.  Every instance launched is terminated at the end,
also after errors or Ctrl+C.  ``--resume STAMP`` follows the instances of an earlier call
(by their geoinv3d:sweep tag) instead of launching new ones; jobs already in OUT_DIR are not
sent again.

Safety: an instance stops itself if no sweep has started within an hour of launch, or
``--timeout-min`` after the sweep started, and 30 minutes after it ended.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoinv3d.cloud.ec2 import REMOTE, EC2Backend, bootstrap_script, code_archive  # noqa: E402

PRICE = {"c5.2xlarge": 0.34, "c5.4xlarge": 0.68, "c5.9xlarge": 1.53, "c6i.4xlarge": 0.68,
         "c6i.8xlarge": 1.36, "m5.4xlarge": 0.77}       # USD per hour, roughly (ap-south-1)
FILES = ("result.zip", "worker.log", "minutes", "exit_code")    # result.json is in the zip too


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def launch(backend: EC2Backend, stamp: str, index: int, instance_type: str) -> str:
    key = backend.ensure_key_pair()
    sg = backend.ensure_security_group()
    ami = backend.latest_ami()
    name = f"geoinv3d-sweep-{stamp}-{index}"
    tags = backend.tags(Name=name, **{"geoinv3d:sweep": stamp})
    resp = backend.ec2.run_instances(
        ImageId=ami["ImageId"], InstanceType=instance_type, MinCount=1, MaxCount=1,
        KeyName=key, SecurityGroupIds=[sg], InstanceInitiatedShutdownBehavior="stop",
        UserData=bootstrap_script(60),
        BlockDeviceMappings=[{"DeviceName": ami.get("RootDeviceName", "/dev/xvda"),
                              "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
        TagSpecifications=[{"ResourceType": t, "Tags": tags}
                           for t in ("instance", "volume", "network-interface")])
    return resp["Instances"][0]["InstanceId"]


def ssh_to(backend: EC2Backend, instance_id: str, timeout_s=900):
    t0 = time.time()
    while True:
        inst = backend._instance(instance_id)
        if inst and inst["State"]["Name"] == "running" and inst.get("PublicIpAddress"):
            try:      # the instance's own key: a --resume call has made a new one for itself
                key = backend.state_dir / "keys" / f"{inst.get('KeyName', backend.key_name)}.pem"
                return backend.ssh_factory(inst["PublicIpAddress"], str(key))
            except Exception:
                pass
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{instance_id}: SSH did not come up")
        time.sleep(10)


def provision(backend, instance_id, jobs: dict, files: list[Path], parallel: int, timeout_min: int):
    """Wait for the set-up, upload the code, the data and the jobs, start the runner."""
    ssh = ssh_to(backend, instance_id)
    try:
        while ssh.read(f"{REMOTE}/READY") is None:
            if ssh.read(f"{REMOTE}/BOOTSTRAP_FAILED") is not None:
                _, out = ssh.run("sudo tail -n 20 /var/log/geoinv3d-bootstrap.log")
                raise RuntimeError(f"{instance_id}: installing the software failed:\n{out}")
            time.sleep(10)
        if ssh.read(f"{REMOTE}/sweep/STARTED") is not None:
            return                       # an earlier call started it: follow it
        ssh.write(f"{REMOTE}/code.tar.gz", code_archive())
        code, out = ssh.run(f"rm -rf {REMOTE}/code/geoinv3d && tar xzf {REMOTE}/code.tar.gz -C {REMOTE}/code "
                            f"&& mkdir -p {REMOTE}/sweep/data {REMOTE}/sweep/out")
        if code:
            raise RuntimeError(f"{instance_id}: unpacking the code failed: {out}")
        for f in files:
            ssh.put(str(f), f"{REMOTE}/sweep/data/{f.name}")
        ssh.write(f"{REMOTE}/sweep/jobs.json", json.dumps(jobs).encode())
        ssh.write(f"{REMOTE}/sweep/sweep_runner.py", (REPO / "deploy" / "sweep_runner.py").read_bytes())
        inner = (f"PYTHONPATH={REMOTE}/code {REMOTE}/venv/bin/python sweep_runner.py jobs.json data out "
                 f"--parallel {parallel} > runner.log 2>&1; sudo shutdown -c; sudo shutdown -h +30")
        code, out = ssh.run(f"cd {REMOTE}/sweep && (sudo shutdown -c || true) && sudo shutdown -h +{timeout_min} "
                            f"&& date +%s > STARTED && {{ nohup setsid bash -c '{inner}' > /dev/null 2>&1 "
                            f"< /dev/null & }} && echo started")
        if code or "started" not in out:
            raise RuntimeError(f"{instance_id}: starting the runner failed: {out}")
    finally:
        ssh.close()


def status(backend, instance_id) -> dict | None:
    ssh = ssh_to(backend, instance_id, timeout_s=60)
    try:
        data = ssh.read(f"{REMOTE}/sweep/out/status.json")
        return json.loads(data) if data else None
    finally:
        ssh.close()


def download(backend, instance_id, out_dir: Path) -> int:
    """The results of every job on the instance: one tar over SSH, unpacked into out_dir."""
    ssh = ssh_to(backend, instance_id, timeout_s=300)
    try:
        names = " -o ".join(f"-name {f}" for f in FILES)
        code, msg = ssh.run(f"cd {REMOTE}/sweep/out && find . -mindepth 2 -maxdepth 2 \\( {names} \\) -print0 "
                            f"| tar czf ../results.tgz --null -T - status.json", timeout=600)
        if code:
            raise RuntimeError(f"{instance_id}: packing the results failed: {msg}")
        data = ssh.read(f"{REMOTE}/sweep/results.tgz")
    finally:
        ssh.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out_dir, filter="data")
    return len(data)


def follow(backend, instance_id, index, jobs, files, args, out_dir, report):
    try:
        if not args.resume:
            provision(backend, instance_id, jobs, files, args.parallel, args.timeout_min)
            log(f"[{index}] {instance_id}: {len(jobs)} jobs started")
        last = None
        while True:
            try:
                s = status(backend, instance_id)
            except Exception as e:      # SSH hiccups: try again
                log(f"[{index}] ! {str(e)[:100]}")
                s = None
            if s:
                line = f"done {s['done']}/{s['total']}, failed {len(s['failed'])}, running {s['running']}"
                if line != last:
                    log(f"[{index}] {line} ({s['minutes']} min)")
                    last = line
                report[index] = s
                if s.get("finished"):
                    break
            time.sleep(30)
        size = download(backend, instance_id, out_dir)
        log(f"[{index}] downloaded {size / 1e6:.1f} MB; terminating {instance_id}")
    except Exception as e:
        log(f"[{index}] failed: {e}")
    finally:
        backend.ec2.terminate_instances(InstanceIds=[instance_id])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--type", default="c5.4xlarge")
    ap.add_argument("--parallel", type=int, default=3, help="jobs at a time on an instance")
    ap.add_argument("--only", default="", help="comma-separated prefixes of the job names")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--timeout-min", type=int, default=6 * 60)
    ap.add_argument("--resume", metavar="STAMP")
    args = ap.parse_args()

    runs = json.loads(Path(args.runs).read_text())
    prefixes = [p for p in args.only.split(",") if p]
    out_dir = Path(args.out)
    names = [n for n in runs if (not prefixes or any(n.startswith(p) for p in prefixes))
             and not ((out_dir / n / "exit_code").exists() and (out_dir / n / "exit_code").read_text().strip() == "0")]
    files = sorted({Path(args.data) / f for n in names for f in runs[n]["files"]})
    backend = EC2Backend(args.region)
    stamp = args.resume or f"sweep-{int(time.time())}"
    if args.resume:
        res = backend.ec2.describe_instances(Filters=[
            {"Name": "tag:geoinv3d:sweep", "Values": [stamp]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]}])
        found = sorted([(next(t["Value"] for t in i["Tags"] if t["Key"] == "Name"), i["InstanceId"])
                        for r in res["Reservations"] for i in r["Instances"]])
        ids = [i for _, i in found]
        shares = [{} for _ in ids]
    else:
        n = min(args.instances, len(names))
        shares = [{name: runs[name]["params"] for name in names[k::n]} for k in range(n)]
        ids = []
    log(f"{stamp}: {len(names)} jobs on {len(shares)} × {args.type}, {args.parallel} at a time; "
        f"owner {backend.owner}, region {args.region}")
    report, threads, t0 = {}, [], time.time()
    try:
        if not args.resume:
            for k in range(len(shares)):
                while True:
                    try:
                        ids.append(launch(backend, stamp, k, args.type))
                        break
                    except Exception as e:
                        if "VcpuLimitExceeded" not in str(e) and "InsufficientInstanceCapacity" not in str(e):
                            raise
                        log(f"[{k}] waiting for capacity: {str(e)[:90]}")
                        time.sleep(60)
                log(f"[{k}] launched {ids[-1]} for {len(shares[k])} jobs")
        for k, (iid, jobs) in enumerate(zip(ids, shares)):
            t = threading.Thread(target=follow, daemon=True,
                                 args=(backend, iid, k, jobs, files, args, out_dir, report))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        if ids:
            backend.ec2.terminate_instances(InstanceIds=ids)
            log(f"terminated {ids} (safety net; harmless if already terminated)")
    hours = (time.time() - t0) / 3600
    done = sum(r.get("done", 0) for r in report.values())
    failed = sorted(f for r in report.values() for f in r.get("failed", []))
    log(f"{done} done, {len(failed)} failed {failed[:10]}; {hours:.2f} h × {len(ids)} instances "
        f"≈ ${hours * len(ids) * PRICE.get(args.type, 0):.2f}")
    return 0 if not failed and done else 1


if __name__ == "__main__":
    sys.exit(main())
