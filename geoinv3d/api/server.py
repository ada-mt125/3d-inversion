"""FastAPI server for GeoInv3D cloud inversion.

Endpoints:
    POST   /api/inversion/submit          — upload data files + params, submit to AWS Batch
    GET    /api/jobs                      — jobs submitted from here: live status, progress,
                                            result summary
    GET    /api/inversion/{job_id}        — one job, refreshed
    GET    /api/inversion/{job_id}/logs   — last lines the worker printed (CloudWatch Logs)
    GET    /api/inversion/{job_id}/result — download the result archive
    DELETE /api/inversion/{job_id}        — stop a job, queued or running
    GET    /api/health

Jobs are remembered in a JSON file (GEOINV3D_JOBS_FILE, default
~/.geoinv3d/jobs.json), so the list survives restarts.  Timestamps are
milliseconds since the epoch (as AWS Batch reports them), except the
worker's progress report, which uses seconds.

The server listens on 127.0.0.1 by default: anyone who can reach it can start
AWS jobs with this machine's AWS credentials.

Usage:
    python -m geoinv3d.api
    python -m geoinv3d.api --port 8000 --bucket my-bucket
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..cloud.aws import AWSRunner, INSTANCE_RESOURCES, TERMINAL_STATUSES

app = FastAPI(
    title="GeoInv3D Inversion API",
    version="0.2.0",
    description="Submit and monitor geophysical inversion jobs on AWS",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CANCEL_REASON = "Cancelled by user"

_runner: Optional[AWSRunner] = None
_store: Optional["JobStore"] = None
_upload_dir = Path(tempfile.gettempdir()) / "geoinv3d_uploads"
_upload_dir.mkdir(exist_ok=True)


def get_runner() -> AWSRunner:
    global _runner
    if _runner is None:
        _runner = AWSRunner(
            bucket=os.environ.get("GEOINV3D_S3_BUCKET", "geoinv3d-data"),
            region=os.environ.get("GEOINV3D_AWS_REGION", "eu-west-1"),
            job_queue=os.environ.get("GEOINV3D_JOB_QUEUE", "geoinv3d-queue"),
            job_definition=os.environ.get("GEOINV3D_JOB_DEF", "geoinv3d-worker"),
        )
    return _runner


class JobStore:
    """Job records kept in a JSON file, keyed by AWS Batch job id."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        try:
            self._jobs: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._jobs = {}
        except json.JSONDecodeError:
            print(f"[API] Ignoring unreadable job list {self.path}")
            self._jobs = {}

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: j.get("submitted_at", 0),
                      reverse=True)

    def put(self, record: dict) -> dict:
        self._jobs[record["job_id"]] = record
        self._save()
        return record

    def update(self, job_id: str, **fields) -> dict:
        record = self._jobs.setdefault(job_id, {"job_id": job_id})
        record.update(fields)
        self._save()
        return record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._jobs, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(os.environ.get(
            "GEOINV3D_JOBS_FILE", Path.home() / ".geoinv3d" / "jobs.json"))
    return _store


def _summarize(result: dict) -> dict:
    """The parts of result.json the job list shows."""
    keys = ("error", "regularization", "n_iterations", "n_data", "n_active_cells",
            "mesh_type", "notes", "outside_core_share", "inversion_mode", "methods")
    summary = {k: result[k] for k in keys if k in result}
    iterations = result.get("iterations") or []
    if iterations:
        summary["final_phi_d"] = iterations[-1].get("phi_d")
    used = (result.get("mesh_design") or {}).get("used")
    if used:
        summary["mesh"] = used
    return summary


def _display_status(record: dict) -> str:
    """Batch status, with FAILED after a user stop shown as CANCELLED."""
    status = record.get("status", "UNKNOWN")
    if status == "FAILED" and (record.get("cancel_requested")
                               or str(record.get("reason", "")).startswith(CANCEL_REASON)):
        return "CANCELLED"
    return status


def _refresh(job_id: str) -> dict:
    """Update a job from AWS: status, the worker's progress and, once finished,
    the result summary.  Finished jobs with a summary are not queried again.
    AWS errors are kept in ``refresh_error`` rather than raised."""
    store = get_store()
    record = store.get(job_id) or {"job_id": job_id}
    if record.get("status") in TERMINAL_STATUSES and "summary" in record:
        return {**record, "display_status": _display_status(record)}
    runner = get_runner()
    fields: dict = {"refreshed_at": int(time.time() * 1000), "refresh_error": None}
    try:
        live = runner.poll(job_id)
        fields.update(live)
        task_id = live.get("task_id") or record.get("task_id")
        if task_id:
            fields["progress"] = runner.progress(task_id)
            if live.get("status") in TERMINAL_STATUSES:
                summary = runner.result_summary(task_id)
                if summary is not None:
                    fields["summary"] = _summarize(summary)
    except Exception as e:
        fields["refresh_error"] = str(e)
    record = store.update(job_id, **fields)
    return {**record, "display_status": _display_status(record)}


@app.post("/api/inversion/submit")
async def submit_inversion(
    files: list[UploadFile] = File(...),
    method: str = Form("gravity"),
    mesh_type: str = Form("tensor"),
    instance_type: str = Form("c5.xlarge"),
    note: str = Form(""),
    params_json: str = Form("{}"),
):
    """Upload data files to S3 and submit an AWS Batch job for them.

    The job asks Batch for the vCPUs and memory of ``instance_type``.
    """
    if instance_type not in INSTANCE_RESOURCES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown instance type '{instance_type}'")
    task_id = uuid.uuid4().hex[:12]
    task_dir = _upload_dir / task_id
    task_dir.mkdir(exist_ok=True)

    saved_files = []
    try:
        for f in files:
            # Keep only the base name: a name like "../x" must not escape task_dir
            name = Path(f.filename or "").name
            if not name:
                raise HTTPException(status_code=400, detail="A file has no name")
            dest = task_dir / name
            with open(dest, "wb") as out:
                out.write(await f.read())
            saved_files.append(str(dest))

        params = json.loads(params_json)
        params.setdefault("method_type", method)
        params.setdefault("mesh_type", mesh_type)

        runner = get_runner()
        actual_task_id, data_prefix = runner.upload_data(saved_files, task_id)
        job_id = runner.submit_pipeline(actual_task_id, data_prefix, params,
                                        instance_type=instance_type)

        get_store().put({
            "job_id": job_id,
            "task_id": actual_task_id,
            "submitted_at": int(time.time() * 1000),
            "method": method,
            "inversion_mode": params.get("inversion_mode"),
            "regularization_type": params.get("regularization_type"),
            "max_iter": params.get("max_iter"),
            "mesh_type": mesh_type,
            "instance_type": instance_type,
            "note": note,
            "status": "SUBMITTED",
            "files": [Path(p).name for p in saved_files],
            "n_files": len(saved_files),
        })

        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            "task_id": actual_task_id,
            "message": f"Job submitted: {len(saved_files)} file(s) uploaded, "
                       f"method={method}, instance={instance_type}",
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


@app.get("/api/jobs")
async def list_jobs(status: str = "all"):
    """Jobs submitted from this machine, newest first, refreshed from AWS."""
    jobs = [_refresh(j["job_id"]) for j in get_store().all()]
    if status != "all":
        jobs = [j for j in jobs if j["display_status"] == status.upper()]
    return JSONResponse({"ok": True, "jobs": jobs, "count": len(jobs)})


@app.get("/api/inversion/{job_id}")
async def get_job_status(job_id: str):
    """One job: Batch status, the worker's progress, and the result summary."""
    return JSONResponse({"ok": True, **_refresh(job_id)})


@app.get("/api/inversion/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 100):
    """The last ``limit`` lines of the worker's log."""
    record = _refresh(job_id)
    stream = record.get("log_stream")
    if not stream:
        return JSONResponse({"ok": True, "lines": [],
                             "message": "No log yet: the job has not started running."})
    try:
        lines = get_runner().tail_logs(stream, limit=max(1, min(limit, 1000)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, "lines": lines})


@app.get("/api/inversion/{job_id}/result")
async def get_job_result(job_id: str):
    """Download the result archive of a finished job."""
    record = get_store().get(job_id) or _refresh(job_id)
    task_id = record.get("task_id")
    if not task_id:
        raise HTTPException(status_code=404, detail="Unknown job")
    try:
        result_dir = _upload_dir / "results"
        result_dir.mkdir(exist_ok=True)
        local_path = get_runner().fetch_result(task_id, str(result_dir))
        return FileResponse(local_path, media_type="application/zip",
                            filename=f"{task_id}_result.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/inversion/{job_id}")
async def cancel_job(job_id: str):
    """Stop a job: a queued job is cancelled, a running one is terminated."""
    try:
        get_runner().cancel(job_id, reason=CANCEL_REASON)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    get_store().update(job_id, cancel_requested=True)
    return JSONResponse({"ok": True, "message": f"Job {job_id} stopped", **_refresh(job_id)})


@app.get("/api/health")
async def health():
    """Health check (does not contact AWS)."""
    has_creds = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or Path.home().joinpath(".aws", "credentials").exists()
    )
    return JSONResponse({
        "ok": True,
        "service": "GeoInv3D Inversion API",
        "aws_configured": has_creds,
        "bucket": os.environ.get("GEOINV3D_S3_BUCKET", "geoinv3d-data"),
        "region": os.environ.get("GEOINV3D_AWS_REGION", "eu-west-1"),
        "job_queue": os.environ.get("GEOINV3D_JOB_QUEUE", "geoinv3d-queue"),
    })


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GeoInv3D API Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bucket", default=None,
                        help="S3 bucket (or set GEOINV3D_S3_BUCKET)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set GEOINV3D_AWS_REGION)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.bucket:
        os.environ["GEOINV3D_S3_BUCKET"] = args.bucket
    if args.region:
        os.environ["GEOINV3D_AWS_REGION"] = args.region

    print(f"Starting GeoInv3D API on {args.host}:{args.port}")
    print(f"  S3 bucket: {os.environ.get('GEOINV3D_S3_BUCKET', 'geoinv3d-data')}")
    print(f"  Region:    {os.environ.get('GEOINV3D_AWS_REGION', 'eu-west-1')}")
    print(f"  Job list:  {get_store().path}")
    print(f"  Docs:      http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "geoinv3d.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
