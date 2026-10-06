"""Local backend: inversion jobs on this machine, run by the same worker as on EC2.

Free and without set-up time, for previews on coarse meshes and for jobs that fit this
machine's memory; large jobs go to AWS (geoinv3d.cloud.ec2, geoinv3d.cloud.aws).

1. ``start_job`` copies the files and params.json to ``<state_dir>/<task_id>/``
   (``data/``, ``params.json``) and queues the job.  At most ``max_jobs`` run at
   once (default 1: a job uses every core); the others wait in the "queued" phase.
2. Each job is a child process, ``python -m geoinv3d.cloud.local <job dir>``, which
   runs the worker's ``run_local_job`` (``out/progress.json`` while it runs, then
   ``out/result.zip`` and ``out/result.json``) and writes ``out/exit_code``.
3. ``refresh`` reads those files, so a job outlives a restart of the API server;
   a process that ended without ``exit_code`` (killed, out of memory) is a failed job.

``cancel`` ends the process; ``request_finish`` writes ``out/STOP``, which the worker
checks after every iteration, as on EC2.  Job records carry a ``phase``: queued,
running, succeeded, failed or cancelled.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..io.filestore import store

PHASE_STATUS = {"queued": "RUNNABLE", "running": "RUNNING", "succeeded": "SUCCEEDED",
                "failed": "FAILED", "cancelled": "FAILED"}
# How often the worker rewrites progress.json here (seconds): often enough for the
# page's live curves, as reading a local file costs nothing
PROGRESS_INTERVAL = 2.0


def total_memory_bytes() -> Optional[int]:
    """This machine's physical memory, or None if it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running."""
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def worker_command(job_dir: Path) -> list[str]:
    """The child process of a job: this module run on the job's folder."""
    return [sys.executable, "-m", "geoinv3d.cloud.local", str(job_dir)]


class LocalBackend:
    """Job backend for the API server (see the module docstring)."""

    def __init__(self, state_dir=None, max_jobs: int = 1,
                 command: Callable[[Path], list[str]] = worker_command,
                 threads: Optional[int] = None, clock=time.time) -> None:
        self.root = Path(state_dir or Path.home() / ".geoinv3d" / "local")
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_jobs = max(1, int(max_jobs))
        self.command, self.threads, self.clock = command, threads, clock
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ── where a job keeps its files ──
    def job_dir(self, record_or_task) -> Path:
        task_id = record_or_task if isinstance(record_or_task, str) else record_or_task["task_id"]
        return self.root / task_id

    def resources(self) -> dict:
        memory = total_memory_bytes()
        return {"cpus": os.cpu_count(), "memory_gb": round(memory / 1e9, 1) if memory else None,
                "max_jobs": self.max_jobs}

    # ── backend interface ──
    def start_job(self, task_id: str, files: list[str], params: dict,
                  instance_type: Optional[str] = None) -> str:
        """Copy the files, queue the job and start it when a slot is free."""
        job = self.job_dir(task_id)
        (job / "data").mkdir(parents=True, exist_ok=True)
        (job / "out").mkdir(exist_ok=True)
        for f in files:      # shares the bytes of an identical input kept before
            store(f, job / "data" / Path(f).name)
        (job / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        (job / "QUEUED").write_text(str(self.clock()), encoding="utf-8")
        self._schedule()
        return f"local-{task_id}"

    def initial_fields(self) -> dict:
        return {"phase": "queued", "status": "RUNNABLE", "backend": "local",
                "created": int(self.clock() * 1000)}

    def _queued(self) -> list[Path]:
        jobs = [p.parent for p in self.root.glob("*/QUEUED")]
        return sorted(jobs, key=lambda j: float((j / "QUEUED").read_text() or 0))

    def _running(self) -> int:
        n = 0
        for started in self.root.glob("*/STARTED"):
            job = started.parent
            if not (job / "out" / "exit_code").exists() and self._alive(job):
                n += 1
        return n

    def _schedule(self) -> None:
        """Start queued jobs, oldest first, while fewer than ``max_jobs`` run."""
        with self._lock:
            free = self.max_jobs - self._running()
            for job in self._queued()[:max(free, 0)]:
                self._start(job)

    def _start(self, job: Path) -> None:
        env = dict(os.environ)
        # the child imports this copy of geoinv3d, wherever the server was started from
        import geoinv3d
        code = str(Path(geoinv3d.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = code + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["TASK_ID"] = job.name
        env["GEOINV3D_PROGRESS_INTERVAL"] = str(PROGRESS_INTERVAL)
        if self.threads:
            for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS"):
                env[k] = str(self.threads)
        log = open(job / "out" / "worker.log", "ab")
        kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
        try:
            proc = subprocess.Popen(self.command(job), cwd=job, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
        finally:
            log.close()
        (job / "STARTED").write_text(json.dumps({"at": self.clock(), "pid": proc.pid}),
                                     encoding="utf-8")
        (job / "QUEUED").unlink(missing_ok=True)
        self._procs[job.name] = proc
        # when it ends, the next queued job starts (also with nobody polling)
        threading.Thread(target=self._watch, args=(proc,), daemon=True).start()

    def _watch(self, proc: subprocess.Popen) -> None:
        proc.wait()
        self._schedule()

    @staticmethod
    def _started(job: Path) -> Optional[dict]:
        try:
            return json.loads((job / "STARTED").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _alive(self, job: Path) -> bool:
        proc = self._procs.get(job.name)
        if proc is not None:
            return proc.poll() is None
        started = self._started(job)   # started before the server last restarted
        return bool(started) and pid_alive(int(started["pid"]))

    def refresh(self, record: dict) -> dict:
        """New fields for a job record: phase, status, progress, times."""
        job, fields = self.job_dir(record), {}
        out = job / "out"
        started = self._started(job)
        if started:
            fields["started"] = int(started["at"] * 1000)
        if (job / "CANCELLED").exists():
            fields.update(phase="cancelled", progress=None, reason="Cancelled by user")
            fields["stopped"] = record.get("stopped") or int((job / "CANCELLED").stat().st_mtime * 1000)
        elif (out / "exit_code").exists():
            code = (out / "exit_code").read_text().strip()
            fields.update(phase="succeeded" if code == "0" else "failed", progress=None)
            fields["stopped"] = record.get("stopped") or int((out / "exit_code").stat().st_mtime * 1000)
            if code != "0":
                fields["reason"] = (self.result_summary(record) or {}).get("error") \
                    or f"The worker exited with status {code}"
        elif started:
            if self._alive(job):
                fields["phase"] = "running"
                try:
                    fields["progress"] = json.loads((out / "progress.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    fields["progress"] = {"stage": "starting"}
            else:
                fields.update(phase="failed", progress=None, stopped=int(self.clock() * 1000),
                              reason="The worker process ended without a result (it may have run "
                                     "out of memory, or the computer restarted)")
        elif (job / "QUEUED").exists():
            self._schedule()   # e.g. after a restart of the server, with nothing running
            if (job / "STARTED").exists():
                return self.refresh(record)
            ahead = [j.name for j in self._queued()].index(job.name)
            fields.update(phase="queued", progress={
                "stage": "queued", "message": "waiting for the job before it to finish" if not ahead
                else f"{ahead + 1} jobs before it"})
        else:
            fields.update(phase="failed", reason="The job's folder on this machine is gone")
        fields["status"] = PHASE_STATUS[fields["phase"]]
        return fields

    def result_summary(self, record: dict) -> Optional[dict]:
        path = self.job_dir(record) / "out" / "result.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def fetch_result(self, record: dict, output_dir=None) -> str:
        path = self.job_dir(record) / "out" / "result.zip"
        if not path.exists():
            raise FileNotFoundError("The result is not there yet")
        return str(path)

    def tail_logs(self, record: dict, limit: int = 100) -> list[str]:
        path = self.job_dir(record) / "out" / "worker.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]

    def cancel(self, record: dict) -> None:
        """Take a queued job off the queue, or end a running one (its result is lost)."""
        job = self.job_dir(record)
        (job / "CANCELLED").write_text("cancelled by the user\n", encoding="utf-8")
        (job / "QUEUED").unlink(missing_ok=True)
        started = self._started(job)
        if started and self._alive(job):
            pid = int(started["pid"])
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   capture_output=True, check=False)
                else:
                    os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        self._schedule()

    def request_finish(self, record: dict) -> None:
        """Ask the worker to stop after its current iteration and keep the result."""
        job = self.job_dir(record)
        if not self._started(job) or not self._alive(job):
            raise RuntimeError("The inversion is not running: there is nothing to stop")
        (job / "out" / "STOP").write_text("stop requested\n", encoding="utf-8")

    def fetch(self, record: dict) -> None:
        raise NotImplementedError("Local jobs keep their result on this machine")


def run(job_dir) -> int:
    """The child process: the worker on the job's files, then ``out/exit_code``."""
    job = Path(job_dir)
    os.chdir(job)
    code = 1
    try:
        from .worker import run_local_job
        code = run_local_job("params.json", "data", "out")
    except BaseException as e:   # the exit code must be written whatever happens
        print(f"[Local] The worker failed: {e!r}", flush=True)
        raise
    finally:
        tmp = job / "out" / "exit_code.tmp"
        tmp.write_text(f"{code}\n", encoding="utf-8")
        os.replace(tmp, job / "out" / "exit_code")
    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m geoinv3d.cloud.local JOB_DIR")
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
