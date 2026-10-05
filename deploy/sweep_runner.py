"""Run many inversion jobs on one machine, a few at a time (the remote half of ec2_sweep.py).

    python deploy/sweep_runner.py jobs.json DATA_DIR OUT_DIR [--parallel 3] [--threads 5]

``jobs.json`` maps a name to the parameters of a job.  Each job runs as
``python -m geoinv3d.cloud.worker --local`` in a process of its own, with ``--threads``
threads, into ``OUT_DIR/<name>/`` (result.zip, result.json, progress.json, worker.log,
exit_code, minutes); a job whose exit_code is already 0 is skipped, so running this again
picks up where an earlier run stopped.  ``OUT_DIR/status.json`` counts the jobs queued,
running, done and failed, and says ``finished`` at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="threads per job (default: cores / parallel)")
    args = ap.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    threads = args.threads or max(1, (os.cpu_count() or 2) // args.parallel)
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           **{k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "NUMBA_NUM_THREADS")}}
    state = {name: "queued" for name in jobs}
    for name in jobs:
        code = out / name / "exit_code"
        if code.exists() and code.read_text().strip() == "0":
            state[name] = "done"
    lock = threading.Lock()
    t0 = time.time()

    def write_status(finished=False):
        with lock:          # one writer at a time: they share the temporary file
            counts = {s: sum(v == s for v in state.values()) for s in ("queued", "running", "done", "failed")}
            status = {**counts, "total": len(jobs), "finished": finished, "threads": threads,
                      "parallel": args.parallel, "minutes": round((time.time() - t0) / 60, 1),
                      "running": [n for n, v in state.items() if v == "running"],
                      "failed": [n for n, v in state.items() if v == "failed"]}
            tmp = out / "status.json.tmp"
            tmp.write_text(json.dumps(status))
            tmp.replace(out / "status.json")

    def run(name):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "params.json").write_text(json.dumps(jobs[name]))
        with lock:
            state[name] = "running"
        write_status()
        start = time.time()
        with open(d / "worker.log", "w") as log:
            try:
                code = subprocess.call([sys.executable, "-m", "geoinv3d.cloud.worker", "--local",
                                        str(d / "params.json"), args.data, str(d)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            except OSError as e:     # a job that cannot start fails alone; the others go on
                log.write(f"could not start: {e}\n")
                code = -1
        (d / "exit_code").write_text(str(code))
        (d / "minutes").write_text(f"{(time.time() - start) / 60:.2f}")
        with lock:
            state[name] = "done" if code == 0 else "failed"
        write_status()

    todo = [n for n, s in state.items() if s == "queued"]
    write_status()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(run, todo))
    write_status(finished=True)
    return 0 if all(v == "done" for v in state.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
