"""FastAPI server for GeoInv3D cloud inversion.

Endpoints:
    GET    /                              — the upload page (geoinv3d/viz/dag_interactive.html)
    POST   /api/inversion/submit          — upload data files + params, start an AWS job
    GET    /api/jobs                      — jobs submitted from here: live status, progress,
                                            result summary
    GET    /api/inversion/{job_id}        — one job, refreshed
    GET    /api/inversion/{job_id}/logs   — last lines the worker printed (CloudWatch Logs)
    GET    /api/inversion/{job_id}/result — download the result archive
    GET    /api/inversion/{job_id}/workflow — the result as a DAG workflow for the viewer
    DELETE /api/inversion/{job_id}        — terminate a job, queued or running (its result is lost)
    POST   /api/inversion/{job_id}/finish — stop a running inversion after this iteration, result kept
    POST   /api/inversion/{job_id}/fetch  — EC2: restart a stopped instance to fetch its result
    GET    /api/workflow?ids=…            — several finished jobs as one workflow
    …      /api/comparisons[/{id}[/page]] — saved comparisons (list, save, open, export, delete)
    …      /api/workspaces[/{id}]         — named workspaces (list, create, open, rename, delete);
                                            jobs submitted with a workspace_id join it
    GET    /api/workspaces/{id}/workflow  — a workspace's finished runs as one workflow
    GET    /api/health

Backends (``backend`` in ~/.geoinv3d/aws.json, or GEOINV3D_BACKEND):
    "ec2" (default) — one EC2 instance per job, driven over SSH; only needs
                      EC2 permissions, tags everything Owner=<IAM user>
                      (geoinv3d.cloud.ec2)
    "batch"         — S3 + AWS Batch (geoinv3d.cloud.aws; see deploy/setup_aws.py)
The region comes from GEOINV3D_AWS_REGION, the config file, or the AWS
profile's default region.

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

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..cloud.aws import (
    AWSRunner, BatchBackend, CANCEL_REASON, INSTANCE_RESOURCES, TERMINAL_STATUSES,
)

app = FastAPI(
    title="GeoInv3D Inversion API",
    version="0.2.0",
    description="Submit and monitor geophysical inversion jobs on AWS",
)

# The server launches EC2 instances with this machine's AWS credentials, so
# only pages on this machine may use it.  CORS alone is not enough: a
# multipart POST is a "simple" request that browsers send without asking
# first, so requests from other origins, and requests to other host names
# (DNS rebinding), are refused outright.  GEOINV3D_ALLOWED_ORIGINS and
# GEOINV3D_ALLOWED_HOSTS (comma-separated) add others.
LOCAL_ORIGIN = re.compile(r"^(null|https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?)$")
# "testserver" is the host name of FastAPI's test client
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1", "testserver"}


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def origin_allowed(origin: str) -> bool:
    """``file://`` pages send the origin "null"; local servers any port."""
    return bool(LOCAL_ORIGIN.match(origin)) or origin in _env_list("GEOINV3D_ALLOWED_ORIGINS")


def host_allowed(host: str) -> bool:
    name = host.rsplit("]", 1)[0] + "]" if host.startswith("[") else host.split(":")[0]
    return name.lower() in LOCAL_HOSTS or name in _env_list("GEOINV3D_ALLOWED_HOSTS")


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN.pattern,
    allow_origins=_env_list("GEOINV3D_ALLOWED_ORIGINS"),
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "")
    if not host_allowed(host):
        return JSONResponse({"detail": f"Host '{host}' is not allowed"}, status_code=403)
    origin = request.headers.get("origin")
    if origin is not None and not origin_allowed(origin):
        return JSONResponse({"detail": f"Requests from {origin} are not allowed"},
                            status_code=403)
    return await call_next(request)


_backend = None
_store: Optional["JobStore"] = None
_upload_dir = Path(tempfile.gettempdir()) / "geoinv3d_uploads"
_upload_dir.mkdir(exist_ok=True)
CONFIG_PATH = Path.home() / ".geoinv3d" / "aws.json"


def load_config() -> dict:
    """~/.geoinv3d/aws.json (or GEOINV3D_AWS_CONFIG), empty if absent."""
    path = Path(os.environ.get("GEOINV3D_AWS_CONFIG", CONFIG_PATH))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def backend_settings() -> dict:
    cfg = load_config()
    region = os.environ.get("GEOINV3D_AWS_REGION") or cfg.get("region")
    if not region:
        import boto3
        region = boto3.session.Session().region_name or "ap-south-1"
    return {
        "backend": os.environ.get("GEOINV3D_BACKEND") or cfg.get("backend", "ec2"),
        "region": region,
        "owner": cfg.get("owner"),
        "bucket": os.environ.get("GEOINV3D_S3_BUCKET") or cfg.get("bucket", "geoinv3d-data"),
        "job_queue": os.environ.get("GEOINV3D_JOB_QUEUE") or cfg.get("job_queue", "geoinv3d-queue"),
        "job_definition": os.environ.get("GEOINV3D_JOB_DEF") or cfg.get("job_definition",
                                                                        "geoinv3d-worker"),
    }


def get_backend():
    global _backend
    if _backend is None:
        cfg = backend_settings()
        if cfg["backend"] == "ec2":
            from ..cloud.ec2 import EC2Backend
            _backend = EC2Backend(cfg["region"], owner=cfg["owner"])
        elif cfg["backend"] == "batch":
            _backend = BatchBackend(AWSRunner(
                bucket=cfg["bucket"], region=cfg["region"], job_queue=cfg["job_queue"],
                job_definition=cfg["job_definition"]))
        else:
            raise ValueError(f"Unknown backend '{cfg['backend']}' (expected 'ec2' or 'batch')")
    return _backend


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
        _write_json(self.path, self._jobs, indent=1)


def _write_json(path: Path, data, **dumps) -> None:
    """Write JSON through a temporary file, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str, **dumps), encoding="utf-8")
    # On Windows a virus scanner or the search indexer can hold the file
    # just written for a moment, and replacing it then fails
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(os.environ.get(
            "GEOINV3D_JOBS_FILE", Path.home() / ".geoinv3d" / "jobs.json"))
    return _store


def _summarize(result: dict) -> dict:
    """The parts of result.json the job list shows."""
    keys = ("error", "regularization", "n_iterations", "n_data", "n_active_cells",
            "mesh_type", "notes", "outside_core_share", "inversion_mode", "methods",
            "converged", "stopped_early")
    summary = {k: result[k] for k in keys if k in result}
    iterations = result.get("iterations") or []
    if iterations:
        summary["final_phi_d"] = iterations[-1].get("phi_d")
    used = (result.get("mesh_design") or {}).get("used")
    if used:
        summary["mesh"] = used
    return summary


def _display_status(record: dict) -> str:
    """Job status, with FAILED after a user stop shown as CANCELLED."""
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
    backend = get_backend()
    fields: dict = {"refreshed_at": int(time.time() * 1000), "refresh_error": None}
    try:
        fields.update(backend.refresh(record))
        if fields.get("status") in TERMINAL_STATUSES:
            summary = backend.result_summary({**record, **fields})
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
    workspace_id: str = Form(""),
):
    """Upload data files to S3 and submit an AWS Batch job for them.

    The job asks Batch for the vCPUs and memory of ``instance_type``.  With
    ``workspace_id`` the job joins that workspace.
    """
    if instance_type not in INSTANCE_RESOURCES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown instance type '{instance_type}'")
    if workspace_id:
        _workspace_path(workspace_id)   # an unknown workspace launches nothing
    # The account's vCPU limit: refuse a job that would exceed it rather than
    # leave it failing at launch
    limit, in_use = vcpu_limit(), vcpus_in_use()
    need = INSTANCE_RESOURCES[instance_type][0]
    if in_use + need > limit:
        raise HTTPException(status_code=409, detail=(
            f"{instance_type} needs {need} vCPUs but {in_use} of the {limit} allowed are in "
            f"use by running jobs; wait for them to finish or choose a smaller instance"))
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
        # Jobs submitted together (e.g. a parameter sweep) share a group
        group = params.pop("group", None) or {}
        params.setdefault("method_type", method)
        params.setdefault("mesh_type", mesh_type)

        params["task_id"] = task_id
        backend = get_backend()
        job_id = backend.start_job(task_id, saved_files, params, instance_type)

        get_store().put({
            **backend.initial_fields(),
            "job_id": job_id,
            "task_id": task_id,
            "submitted_at": int(time.time() * 1000),
            "method": method,
            "inversion_mode": params.get("inversion_mode"),
            "regularization_type": params.get("regularization_type"),
            "max_iter": params.get("max_iter"),
            "mesh_type": mesh_type,
            "instance_type": instance_type,
            "note": note,
            "files": [Path(p).name for p in saved_files],
            "n_files": len(saved_files),
            "group_id": group.get("id"),
            "group_label": group.get("label"),
            "variant": group.get("variant"),
            "workspace_id": workspace_id or None,
        })
        if workspace_id:
            ws = _read_workspace(workspace_id)
            ws["job_ids"].append(job_id)
            _write_json(_workspace_path(workspace_id), ws)

        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            "task_id": task_id,
            "message": f"Job submitted: {len(saved_files)} file(s) uploaded, "
                       f"method={method}, instance={instance_type}",
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def vcpu_limit() -> int:
    """vCPUs the account may run at once (GEOINV3D_VCPU_LIMIT, aws.json "vcpu_limit")."""
    return int(os.environ.get("GEOINV3D_VCPU_LIMIT") or load_config().get("vcpu_limit", 36))


def vcpus_in_use() -> int:
    """vCPUs of the jobs that may still hold an instance."""
    total = 0
    for record in get_store().all():
        if record.get("status") in TERMINAL_STATUSES or record.get("status") == "STOPPED":
            continue
        status = _refresh(record["job_id"]).get("status")
        if status not in TERMINAL_STATUSES and status != "STOPPED":
            total += INSTANCE_RESOURCES.get(record.get("instance_type"), (0, 0))[0]
    return total


def _run_label(record: dict) -> str:
    """How a job is named in a workflow: its variant, note or settings."""
    parts = [record.get("variant"), record.get("note")]
    label = " · ".join(p for p in parts if p)
    return label or f"{record.get('regularization_type') or 'inversion'} · {record['job_id']}"


def _workflow_for(job_ids: list[str]) -> dict:
    """One DAG workflow holding the results of several finished jobs."""
    from ..viz.result_workflow import build_workflow, load_result

    runs = []
    result_dir = _upload_dir / "results"
    result_dir.mkdir(exist_ok=True)
    for job_id in job_ids:
        record = get_store().get(job_id) or _refresh(job_id)
        if not record.get("task_id"):
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
        run = load_result(get_backend().fetch_result(record, str(result_dir)))
        run["_name"] = _run_label(record)
        run["_backend"] = record.get("backend", "ec2")
        runs.append(run)
    return build_workflow(runs)


@app.get("/api/jobs")
async def list_jobs(status: str = "all", workspace: str = ""):
    """Jobs submitted from this machine (or one workspace), newest first, refreshed from AWS."""
    records = get_store().all()
    if workspace:
        records = [j for j in records if j.get("workspace_id") == workspace]
    jobs = [_refresh(j["job_id"]) for j in records]
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
    try:
        lines = get_backend().tail_logs(record, limit=max(1, min(limit, 1000)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not lines:
        return JSONResponse({"ok": True, "lines": [],
                             "message": "No log yet: the job has not started running."})
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
        local_path = get_backend().fetch_result(record, str(result_dir))
        return FileResponse(local_path, media_type="application/zip",
                            filename=f"{task_id}_result.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/inversion/{job_id}/workflow")
async def get_job_workflow(job_id: str):
    """A finished job as a DAG workflow (mesh, survey, inversion with its
    convergence, 3D model and data fit) that the viewer loads directly."""
    from ..viz.result_workflow import build_workflow, load_result

    record = get_store().get(job_id) or _refresh(job_id)
    if not record.get("task_id"):
        raise HTTPException(status_code=404, detail="Unknown job")
    try:
        result_dir = _upload_dir / "results"
        result_dir.mkdir(exist_ok=True)
        run = load_result(get_backend().fetch_result(record, str(result_dir)))
        run["_name"] = _run_label(record)
        return JSONResponse(build_workflow([run]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Comparing jobs: several results in one workflow, saved for later ──
COMPARISONS_DIR = Path.home() / ".geoinv3d" / "comparisons"


def _comparisons_dir() -> Path:
    path = Path(os.environ.get("GEOINV3D_COMPARISONS_DIR", COMPARISONS_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _comparison_path(comparison_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", comparison_id):
        raise HTTPException(status_code=404, detail="Unknown comparison")
    path = _comparisons_dir() / f"{comparison_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown comparison")
    return path


def _job_ids(ids) -> list[str]:
    job_ids = [i.strip() for i in (ids.split(",") if isinstance(ids, str) else ids or [])
               if str(i).strip()]
    if not job_ids:
        raise HTTPException(status_code=400, detail="List at least one job")
    if len(set(job_ids)) != len(job_ids):
        raise HTTPException(status_code=400, detail="A job is listed twice")
    return job_ids


@app.get("/api/workflow")
async def compare_jobs(ids: str):
    """Finished jobs (comma-separated ids) as one DAG workflow."""
    try:
        return JSONResponse(_workflow_for(_job_ids(ids)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/comparisons")
async def save_comparison(body: dict = Body(...)):
    """Save jobs' results as one named workflow: {"name": ..., "job_ids": [...]}."""
    job_ids = _job_ids(body.get("job_ids"))
    name = str(body.get("name") or "").strip() or f"{len(job_ids)} jobs"
    try:
        workflow = _workflow_for(job_ids)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    comparison_id = uuid.uuid4().hex[:12]
    meta = {"id": comparison_id, "name": name[:200], "job_ids": job_ids,
            "created_at": int(time.time() * 1000)}
    workflow["comparison"] = meta
    _comparisons_dir().joinpath(f"{comparison_id}.json").write_text(
        json.dumps(workflow, separators=(",", ":")), encoding="utf-8")
    return JSONResponse({"ok": True, **meta})


@app.get("/api/comparisons")
async def list_comparisons():
    """Saved comparisons, newest first."""
    items = []
    for path in _comparisons_dir().glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8")).get("comparison")
        except (OSError, json.JSONDecodeError):
            continue
        if meta:
            items.append(meta)
    items.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return JSONResponse({"ok": True, "comparisons": items})


@app.get("/api/comparisons/{comparison_id}")
async def get_comparison(comparison_id: str):
    return FileResponse(_comparison_path(comparison_id), media_type="application/json")


@app.get("/api/comparisons/{comparison_id}/page")
async def comparison_page(comparison_id: str):
    """The comparison as a stand-alone viewer page (open it without the server)."""
    from ..viz.serve_dag import generate_viewer
    path = _comparison_path(comparison_id)
    name = json.loads(path.read_text(encoding="utf-8"))["comparison"]["name"]
    out = _upload_dir / f"comparison_{comparison_id}.html"
    generate_viewer(str(path), str(out))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or comparison_id
    return FileResponse(out, media_type="text/html", filename=f"{safe}.geoinv3d_viewer.html")


@app.delete("/api/comparisons/{comparison_id}")
async def delete_comparison(comparison_id: str):
    _comparison_path(comparison_id).unlink()
    return JSONResponse({"ok": True})


# ── Workspaces: a named viewer whose uploads and results stay together ──
# A workspace is {id, name, created_at, opened_at, job_ids}.  Jobs submitted from a
# page showing a workspace join it, and its workflow holds all of them that finished
# (runs on the same data share one data node; see result_workflow.build_workflow).
WORKSPACES_DIR = Path.home() / ".geoinv3d" / "workspaces"


def _workspaces_dir() -> Path:
    path = Path(os.environ.get("GEOINV3D_WORKSPACES_DIR", WORKSPACES_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workspace_path(workspace_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", workspace_id or ""):
        raise HTTPException(status_code=404, detail="Unknown workspace")
    path = _workspaces_dir() / f"{workspace_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown workspace")
    return path


def _read_workspace(workspace_id: str) -> dict:
    return json.loads(_workspace_path(workspace_id).read_text(encoding="utf-8"))


def _workspace_name(name) -> str:
    name = " ".join(str(name or "").split())
    if len(name) > 120:
        raise HTTPException(status_code=400, detail="Names are at most 120 characters")
    return name or "Untitled workspace"


def _workspace_jobs(ws: dict) -> list[dict]:
    """The workspace's jobs, refreshed, in the order they were submitted."""
    return [_refresh(job_id) for job_id in ws["job_ids"] if get_store().get(job_id)]


def _running(job: dict) -> bool:
    return job.get("status") not in TERMINAL_STATUSES and job.get("status") != "STOPPED"


@app.get("/api/workspaces")
async def list_workspaces():
    """Workspaces, the one opened last first."""
    items = []
    for path in _workspaces_dir().glob("*.json"):
        try:
            ws = json.loads(path.read_text(encoding="utf-8"))
            items.append({k: ws.get(k) for k in ("id", "name", "created_at", "opened_at")}
                         | {"n_jobs": len(ws.get("job_ids", []))})
        except (OSError, ValueError):
            continue
    items.sort(key=lambda w: w.get("opened_at") or 0, reverse=True)
    return JSONResponse({"ok": True, "workspaces": items})


@app.post("/api/workspaces")
async def create_workspace(body: dict = Body(default={})):
    now = int(time.time() * 1000)
    ws = {"id": uuid.uuid4().hex[:12], "name": _workspace_name(body.get("name")),
          "created_at": now, "opened_at": now, "job_ids": []}
    _write_json(_workspaces_dir() / f"{ws['id']}.json", ws)
    return JSONResponse({"ok": True, **ws})


@app.get("/api/workspaces/{workspace_id}")
async def open_workspace(workspace_id: str):
    """A workspace and its jobs; it becomes the one opened last."""
    ws = _read_workspace(workspace_id)
    ws["opened_at"] = int(time.time() * 1000)
    _write_json(_workspace_path(workspace_id), ws)
    return JSONResponse({"ok": True, **ws, "jobs": _workspace_jobs(ws)})


@app.patch("/api/workspaces/{workspace_id}")
async def rename_workspace(workspace_id: str, body: dict = Body(...)):
    ws = _read_workspace(workspace_id)
    ws["name"] = _workspace_name(body.get("name"))
    _write_json(_workspace_path(workspace_id), ws)
    return JSONResponse({"ok": True, **ws})


@app.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Forget a workspace; its jobs and their results are kept."""
    _workspace_path(workspace_id).unlink()
    (_workspaces_dir() / "cache" / f"{workspace_id}.json").unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/workspaces/{workspace_id}/workflow")
async def workspace_workflow(workspace_id: str, known: str = ""):
    """The workspace's finished runs as one workflow.

    ``key`` names the set of finished runs.  With ``known`` equal to it only the
    counts come back (the page polls this); the workflow itself is rebuilt only when
    a run has finished since, and cached until then.
    """
    ws = _read_workspace(workspace_id)
    jobs = _workspace_jobs(ws)
    done = [j["job_id"] for j in jobs if j.get("display_status") == "SUCCEEDED"]
    key = hashlib.sha1(",".join(done).encode()).hexdigest()[:12] if done else ""
    counts = {"key": key, "finished": len(done), "running": sum(map(_running, jobs)), "jobs": len(jobs)}
    if known and known == key:
        return JSONResponse({"ok": True, "unchanged": True, **counts})
    if not done:
        return JSONResponse({"ok": True, **counts, "workflow": None})
    cache = _workspaces_dir() / "cache" / f"{workspace_id}.json"
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return JSONResponse({"ok": True, **counts, "workflow": cached["workflow"]})
    except (OSError, ValueError):
        pass
    try:
        workflow = _workflow_for(done)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    _write_json(cache, {"key": key, "workflow": workflow}, separators=(",", ":"))
    return JSONResponse({"ok": True, **counts, "workflow": workflow})


@app.delete("/api/inversion/{job_id}")
async def cancel_job(job_id: str):
    """Stop a job: a queued job is cancelled, a running one is terminated."""
    record = get_store().get(job_id) or {"job_id": job_id}
    try:
        get_backend().cancel(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    get_store().update(job_id, cancel_requested=True)
    return JSONResponse({"ok": True, "message": f"Job {job_id} stopped", **_refresh(job_id)})


@app.post("/api/inversion/{job_id}/finish")
async def finish_job(job_id: str):
    """Stop a running inversion after its current iteration and keep its result.

    Unlike DELETE (which terminates the instance), the worker ends the
    inversion, packs the model it has, and the result is collected as usual.
    """
    if not get_store().get(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    record = _refresh(job_id)
    if record.get("display_status") != "RUNNING":
        raise HTTPException(status_code=409, detail="Only a running inversion can be stopped "
                                                    "with its result kept")
    try:
        get_backend().request_finish(record)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    get_store().update(job_id, finish_requested=int(time.time() * 1000))
    return JSONResponse({"ok": True, **_refresh(job_id)})


@app.post("/api/inversion/{job_id}/fetch")
async def fetch_stopped_job(job_id: str):
    """EC2 backend: start a stopped instance again and collect its result."""
    record = get_store().get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown job")
    try:
        get_backend().fetch(record)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, **_refresh(job_id)})


PAGE_PATH = Path(__file__).resolve().parent.parent / "viz" / "dag_interactive.html"


@app.get("/", include_in_schema=False)
async def upload_page():
    """The upload page, served from here so it talks to this server whatever the port."""
    page = PAGE_PATH.read_text(encoding="utf-8")
    page = page.replace("<head>", "<head>\n<script>window.GEOINV3D_API = location.origin;</script>", 1)
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health():
    """Health check (does not contact AWS)."""
    has_creds = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or Path.home().joinpath(".aws", "credentials").exists()
    )
    cfg = backend_settings()
    return JSONResponse({
        "ok": True,
        "service": "GeoInv3D Inversion API",
        "aws_configured": has_creds,
        "backend": cfg["backend"],
        "region": cfg["region"],
    })


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GeoInv3D API Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ec2", "batch"], default=None,
                        help="Job backend (or set GEOINV3D_BACKEND; default ec2)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set GEOINV3D_AWS_REGION)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.backend:
        os.environ["GEOINV3D_BACKEND"] = args.backend
    if args.region:
        os.environ["GEOINV3D_AWS_REGION"] = args.region

    cfg = backend_settings()
    print(f"Starting GeoInv3D API on {args.host}:{args.port}")
    print(f"  Backend:   {cfg['backend']}")
    print(f"  Region:    {cfg['region']}")
    print(f"  Job list:  {get_store().path}")
    print(f"  Page:      http://{args.host}:{args.port}/")
    print(f"  Docs:      http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "geoinv3d.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
