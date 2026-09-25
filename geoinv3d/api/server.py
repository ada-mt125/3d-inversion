"""FastAPI server for GeoInv3D cloud inversion.

Endpoints:
    POST /api/inversion/submit  — upload data files + params, submit to AWS
    GET  /api/inversion/{job_id} — poll job status
    GET  /api/inversion/{job_id}/result — download result archive
    GET  /api/jobs               — list recent jobs
    DELETE /api/inversion/{job_id} — cancel a job

Usage:
    python -m geoinv3d.api.server
    python -m geoinv3d.api.server --port 8000 --bucket my-bucket
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..cloud.aws import AWSRunner

app = FastAPI(
    title="GeoInv3D Inversion API",
    version="0.1.0",
    description="Submit and monitor geophysical inversion jobs on AWS",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_runner: Optional[AWSRunner] = None
_jobs: dict[str, dict] = {}
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


@app.post("/api/inversion/submit")
async def submit_inversion(
    files: list[UploadFile] = File(...),
    method: str = Form("gravity"),
    mesh_type: str = Form("tensor"),
    instance_type: str = Form("c5.xlarge"),
    note: str = Form(""),
    params_json: str = Form("{}"),
):
    """Upload data files and submit an inversion job to AWS.

    The files are uploaded to S3, then an AWS Batch job is submitted
    with the specified parameters.
    """
    task_id = uuid.uuid4().hex[:12]
    task_dir = _upload_dir / task_id
    task_dir.mkdir(exist_ok=True)

    saved_files = []
    try:
        for f in files:
            dest = task_dir / f.filename
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)
            saved_files.append(str(dest))

        params = json.loads(params_json)
        params.setdefault("method_type", method)
        params.setdefault("mesh_type", mesh_type)

        runner = get_runner()
        actual_task_id, data_prefix = runner.upload_data(saved_files, task_id)
        job_id = runner.submit_pipeline(actual_task_id, data_prefix, params)

        job_record = {
            "job_id": job_id,
            "task_id": actual_task_id,
            "method": method,
            "mesh_type": mesh_type,
            "instance_type": instance_type,
            "note": note,
            "status": "SUBMITTED",
            "files": [f.filename for f in files],
            "n_files": len(files),
        }
        _jobs[job_id] = job_record

        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            "task_id": actual_task_id,
            "message": f"Job submitted: {len(files)} file(s) uploaded, "
                       f"method={method}, instance={instance_type}",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


@app.get("/api/inversion/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of a submitted job."""
    try:
        runner = get_runner()
        status = runner.poll(job_id)

        if job_id in _jobs:
            _jobs[job_id]["status"] = status.get("status", "UNKNOWN")

        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            **status,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/inversion/{job_id}/result")
async def get_job_result(job_id: str):
    """Download the result archive for a completed job."""
    record = _jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found in local cache")

    task_id = record["task_id"]
    try:
        runner = get_runner()
        result_dir = _upload_dir / "results"
        result_dir.mkdir(exist_ok=True)
        local_path = runner.fetch_result(task_id, str(result_dir))
        return FileResponse(
            local_path,
            media_type="application/zip",
            filename=f"{task_id}_result.zip",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def list_jobs(status: str = "all"):
    """List jobs tracked by this server instance."""
    if status == "all":
        jobs = list(_jobs.values())
    else:
        jobs = [j for j in _jobs.values() if j["status"] == status.upper()]

    for job in jobs:
        try:
            runner = get_runner()
            live = runner.poll(job["job_id"])
            job["status"] = live.get("status", job["status"])
        except Exception:
            pass

    return JSONResponse({"ok": True, "jobs": jobs, "count": len(jobs)})


@app.delete("/api/inversion/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running or pending job."""
    try:
        runner = get_runner()
        runner.cancel(job_id)
        if job_id in _jobs:
            _jobs[job_id]["status"] = "CANCELLED"
        return JSONResponse({"ok": True, "message": f"Job {job_id} cancelled"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check."""
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
    })


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GeoInv3D API Server")
    parser.add_argument("--host", default="0.0.0.0")
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
    print(f"  Docs:      http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "geoinv3d.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
