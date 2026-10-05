"""FastAPI server for GeoInv3D cloud inversion.

Endpoints:
    GET    /                              — the upload page (geoinv3d/viz/dag_interactive.html)
    POST   /api/inversion/submit          — upload data files + params, start a job on this
                                            machine (backend=local) or on AWS (backend=aws)
    GET    /api/inversion/{job_id}/params — the parameters and input files a job ran with
    POST   /api/inversion/{job_id}/rerun  — run a job again on its own inputs, with changes
                                            (or a preview's full-resolution settings)
    GET    /api/inversion/{job_id}/inputs/{name} — one of a job's input files
    GET    /api/inversion/{job_id}/bundle — a job's inputs as a zip (params.json, data/), to
                                            run it again anywhere: geoinv3d.cloud.worker --local
    GET    /api/estimates                 — run times of finished jobs, to estimate new ones
    POST   /api/mesh/cells                — the cells a job's mesh would have (built as the worker does)
    POST   /api/dem, GET /api/dem/{id}    — a DEM downloaded for a box (SRTM, else ETOPO 2022)
    GET    /api/igrf                      — the IGRF-14 field at a place and date
    GET    /api/crs                       — check a coordinate system (e.g. EPSG:32643)
    POST   /api/enhance                   — maps of a data file: reduced to the pole, derivatives,
                                            edge detectors, a regional-residual separation
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
    GET/PUT /api/workspaces/{id}/layers   — a workspace's map layers (points, outlines)
    GET    /api/health                    — the backends: this machine's cores and memory, AWS

Backends: every job runs either on this machine or on AWS, as chosen when it is submitted.
    "local"         — a child process on this machine, queued so that at most
                      GEOINV3D_LOCAL_JOBS (aws.json "local_jobs", default 1) run at once
                      (geoinv3d.cloud.local); needs no AWS account
The AWS backend (``backend`` in ~/.geoinv3d/aws.json, or GEOINV3D_BACKEND):
    "ec2" (default) — one EC2 instance per job, driven over SSH; only needs
                      EC2 permissions, tags everything Owner=<IAM user>
                      (geoinv3d.cloud.ec2)
    "batch"         — S3 + AWS Batch (geoinv3d.cloud.aws; see deploy/setup_aws.py)
The region comes from GEOINV3D_AWS_REGION, the config file, or the AWS
profile's default region.  A job's input files and parameters are kept
(GEOINV3D_INPUTS_DIR, default ~/.geoinv3d/inputs) so that it can be run again.

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
from functools import lru_cache
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
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


_backend = None           # the AWS backend, made when an AWS job first needs it
_local_backend = None
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


def get_local_backend():
    """The backend of jobs run on this machine."""
    global _local_backend
    if _local_backend is None:
        from ..cloud.local import LocalBackend
        _local_backend = LocalBackend(
            os.environ.get("GEOINV3D_LOCAL_DIR") or Path.home() / ".geoinv3d" / "local",
            max_jobs=int(os.environ.get("GEOINV3D_LOCAL_JOBS") or load_config().get("local_jobs", 1)))
    return _local_backend


def backend_for(record: dict):
    """The backend a job runs on: this machine's, or the AWS one."""
    return get_local_backend() if record.get("backend") == "local" else get_backend()


def get_backend():
    """The AWS backend (ec2 or batch)."""
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

    def delete(self, job_id: str) -> Optional[dict]:
        record = self._jobs.pop(job_id, None)
        if record is not None:
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
            "converged", "stopped_early", "convergence")
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
        return {**record, "display_status": _display_status(record), "rerun": _has_inputs(record)}
    backend = backend_for(record)
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
    return {**record, "display_status": _display_status(record), "rerun": _has_inputs(record)}


@app.post("/api/inversion/submit")
async def submit_inversion(
    files: list[UploadFile] = File(...),
    method: str = Form("gravity"),
    mesh_type: str = Form("tensor"),
    instance_type: str = Form("c5.xlarge"),
    note: str = Form(""),
    params_json: str = Form("{}"),
    workspace_id: str = Form(""),
    backend: str = Form("aws"),
):
    """Upload data files and start a job for them, on this machine or on AWS.

    ``backend`` is "local" (this machine; ``instance_type`` is not used) or "aws" (the
    configured AWS backend, on an instance of ``instance_type``).  With ``workspace_id``
    the job joins that workspace.  ``params_json`` may carry ``preview_of``:
    {"params", "backend", "instance_type"} of the full-resolution run that this job
    previews, kept for ``/api/inversion/{job_id}/rerun`` with ``full``.
    """
    task_id = uuid.uuid4().hex[:12]
    task_dir = _upload_dir / task_id
    task_dir.mkdir(exist_ok=True)
    try:
        saved_files = []
        for f in files:
            # Keep only the base name: a name like "../x" must not escape task_dir
            name = Path(f.filename or "").name
            if not name:
                raise HTTPException(status_code=400, detail="A file has no name")
            dest = task_dir / name
            with open(dest, "wb") as out:
                out.write(await f.read())
            saved_files.append(dest)
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"params_json is not JSON: {e}")
        params.setdefault("method_type", method)
        params.setdefault("mesh_type", mesh_type)
        return JSONResponse(_launch(task_id, saved_files, params, method=method,
                                    mesh_type=mesh_type, instance_type=instance_type, note=note,
                                    workspace_id=workspace_id, backend=backend))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


BACKENDS = ("local", "aws")


def _launch(task_id: str, files: list[Path], params: dict, *, method: str, mesh_type: str,
            instance_type: str, note: str, workspace_id: str, backend: str,
            extra: Optional[dict] = None) -> dict:
    """Start a job: keep its inputs, hand it to its backend and record it."""
    if backend not in BACKENDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown backend '{backend}' (expected 'local' or 'aws')")
    local = backend == "local"
    if not local and instance_type not in INSTANCE_RESOURCES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown instance type '{instance_type}'")
    if workspace_id:
        _workspace_path(workspace_id)   # an unknown workspace launches nothing
    if not local:
        # The account's vCPU limit: refuse a job that would exceed it rather than
        # leave it failing at launch
        limit, in_use = vcpu_limit(), vcpus_in_use()
        need = INSTANCE_RESOURCES[instance_type][0]
        if in_use + need > limit:
            raise HTTPException(status_code=409, detail=(
                f"{instance_type} needs {need} vCPUs but {in_use} of the {limit} allowed are in "
                f"use by running jobs; wait for them to finish or choose a smaller instance"))
    params = dict(params)
    # Jobs submitted together (e.g. a parameter sweep) share a group
    group = params.pop("group", None) or {}
    preview_of = params.pop("preview_of", None)
    if preview_of is not None and not isinstance((preview_of or {}).get("params"), dict):
        raise HTTPException(status_code=400, detail="preview_of needs the full run's params")
    params["task_id"] = task_id
    _keep_inputs(task_id, files, params, preview_of)
    try:
        runner = get_local_backend() if local else get_backend()
        job_id = runner.start_job(task_id, [str(f) for f in files], params,
                                  None if local else instance_type)
    except Exception as e:
        shutil.rmtree(_inputs_dir(task_id), ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    record = {
        **runner.initial_fields(),
        "job_id": job_id,
        "task_id": task_id,
        "submitted_at": int(time.time() * 1000),
        "method": method,
        "inversion_mode": params.get("inversion_mode"),
        "regularization_type": params.get("regularization_type"),
        "max_iter": params.get("max_iter"),
        "mesh_type": mesh_type,
        "instance_type": "local" if local else instance_type,
        "note": note,
        "files": [Path(p).name for p in files],
        "n_files": len(files),
        "group_id": group.get("id"),
        "group_label": group.get("label"),
        "variant": group.get("variant"),
        "workspace_id": workspace_id or None,
        "preview": bool(preview_of),
        **(extra or {}),
    }
    if local:
        record["backend"] = "local"
    get_store().put(record)
    if workspace_id:
        ws = _read_workspace(workspace_id)
        ws["job_ids"].append(job_id)
        _write_json(_workspace_path(workspace_id), ws)
    where = "this machine" if local else instance_type
    return {"ok": True, "job_id": job_id, "task_id": task_id,
            "message": f"Job submitted: {len(files)} file(s), method={method}, on {where}"}


# ── A job's inputs, kept so that it can be run again (with changes, or at full resolution) ──
INPUTS_DIR = Path.home() / ".geoinv3d" / "inputs"


def _inputs_dir(task_id: str) -> Path:
    return Path(os.environ.get("GEOINV3D_INPUTS_DIR", INPUTS_DIR)) / task_id


def _keep_inputs(task_id: str, files: list[Path], params: dict, preview_of) -> None:
    folder = _inputs_dir(task_id)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, folder / "data" / Path(f).name)
    _write_json(folder / "params.json", params, indent=1)
    if preview_of:
        _write_json(folder / "preview_of.json", preview_of, indent=1)


def _has_inputs(record: dict) -> bool:
    return bool(record.get("task_id")) and (_inputs_dir(record["task_id"]) / "params.json").exists()


def _job_inputs(job_id: str) -> tuple[dict, Path, dict, Optional[dict]]:
    """A job's record, input folder, parameters and (for a preview) the full run's
    {"params", "backend", "instance_type"}."""
    record = get_store().get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown job")
    folder = _inputs_dir(record["task_id"])
    if not (folder / "params.json").exists():
        raise HTTPException(status_code=404, detail=(
            "This job's inputs were not kept (it was submitted before they were): submit it "
            "again from the upload page"))
    params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    full = folder / "preview_of.json"
    return record, folder, params, (json.loads(full.read_text(encoding="utf-8"))
                                    if full.exists() else None)


# Keys the worker reads only in manual mode (geoinv3d.cloud.worker._MANUAL_KEYS): changing
# one in an auto job makes it a manual one, with the worker's defaults for the rest
def _manual_keys() -> tuple:
    from ..cloud.worker import _MANUAL_KEYS
    return _MANUAL_KEYS


def merge_params(base: dict, changes: dict) -> dict:
    """``base`` with ``changes`` applied: dicts merge key by key, a list of dicts of the
    same length merges item by item (e.g. one dataset's noise floor), anything else
    (and None, which removes a key) replaces."""
    out = dict(base)
    for key, value in changes.items():
        old = out.get(key)
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(old, dict):
            out[key] = merge_params(old, value)
        elif (isinstance(value, list) and isinstance(old, list) and len(value) == len(old)
              and all(isinstance(v, dict) for v in value) and all(isinstance(o, dict) for o in old)):
            out[key] = [merge_params(o, v) for o, v in zip(old, value)]
        else:
            out[key] = value
    return out


@app.get("/api/inversion/{job_id}/params")
async def job_params(job_id: str):
    """The parameters a job ran with, its input files, and (for a preview) the
    settings of the full-resolution run."""
    record, folder, params, full = _job_inputs(job_id)
    files = sorted(p.name for p in (folder / "data").iterdir())
    return JSONResponse({"ok": True, "job_id": job_id, "params": params, "files": files,
                         "preview_of": full, "backend": record.get("backend"),
                         "instance_type": record.get("instance_type"), "note": record.get("note")})


@app.get("/api/inversion/{job_id}/inputs/{name}")
async def job_input(job_id: str, name: str):
    """One input file of a job (the page loads a job's files to start a new setup from it)."""
    _, folder, _, _ = _job_inputs(job_id)
    path = folder / "data" / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No input file '{name}'")
    return FileResponse(path, filename=path.name)


BUNDLE_README = """GeoInv3D job {job_id}: its parameters and input files.

Run it again on any machine with geoinv3d installed (pip install -e . from the repository):

    python -m geoinv3d.cloud.worker --local params.json data out

(or, without installing, with the repository on the path:
    PYTHONPATH=/path/to/3d-inversion python -m geoinv3d.cloud.worker --local params.json data out)

out/ then holds progress.json while it runs, and result.zip and result.json at the end;
open result.zip in the viewer (python -m geoinv3d.viz.serve_dag) or load it on the upload page.
Edit params.json to change the settings; see README.md of the repository for the keys.
"""


@app.get("/api/inversion/{job_id}/bundle")
async def job_bundle(job_id: str):
    """A job's inputs as one zip: params.json, data/ and a README on running it."""
    import zipfile
    record, folder, params, _ = _job_inputs(job_id)
    out = _upload_dir / f"bundle_{record['task_id']}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("params.json", json.dumps(params, indent=1))
        zf.writestr("README.txt", BUNDLE_README.format(job_id=job_id))
        for f in sorted((folder / "data").iterdir()):
            zf.write(f, f"data/{f.name}")
    return FileResponse(out, media_type="application/zip",
                        filename=f"{record['task_id']}_inputs.zip")


@app.post("/api/inversion/{job_id}/rerun")
async def rerun_job(job_id: str, body: dict = Body(default={})):
    """Run a job again on its own input files.

    Body (all optional): ``params`` — the whole set of parameters to run with (default the
    job's own); ``changes`` — parameters to change on top (see :func:`merge_params`);
    ``full`` — start from the full-resolution run the job previewed; ``backend``
    ("local" or "aws", default the job's own, or the full run's), ``instance_type``,
    ``note``, ``workspace_id`` (default the job's own).  Changing a manual-mode setting
    of an auto job makes it a manual one.  The new job records which job it came
    from and what changed.
    """
    record, folder, base, full = _job_inputs(job_id)
    target = {}
    if body.get("full"):
        if not full:
            raise HTTPException(status_code=400, detail="This job is not a preview")
        base, target = full["params"], full
    base = {k: v for k, v in base.items() if k not in ("task_id", "group", "preview_of")}
    params = body.get("params") if body.get("params") is not None else base
    changes = body.get("changes") or {}
    if not isinstance(params, dict) or not isinstance(changes, dict):
        raise HTTPException(status_code=400, detail="params and changes must be objects")
    params = merge_params({k: v for k, v in params.items()
                           if k not in ("task_id", "group", "preview_of")}, changes)
    for key in ("method_type", "mesh_type"):   # what the job is, not a setting to edit
        if key in base:
            params.setdefault(key, base[key])
    changed = {k: params.get(k) for k in sorted(set(base) | set(params))
               if base.get(k) != params.get(k)}
    if base.get("param_mode") == "auto" and params.get("param_mode", "auto") == "auto" \
            and any(k in changed for k in _manual_keys()):
        params["param_mode"] = changed["param_mode"] = "manual"
    backend = body.get("backend") or target.get("backend") or (
        "local" if record.get("backend") == "local" else "aws")
    instance_type = body.get("instance_type") or target.get("instance_type") or (
        record.get("instance_type") if record.get("instance_type") in INSTANCE_RESOURCES
        else "c5.2xlarge")
    workspace_id = body.get("workspace_id", record.get("workspace_id")) or ""
    files = sorted((folder / "data").iterdir())
    note = body.get("note")
    if note is None:
        note = ("full resolution of " if body.get("full") else "re-run of ") + \
            (record.get("note") or job_id)
    return JSONResponse(_launch(
        uuid.uuid4().hex[:12], files, params, method=record.get("method") or params.get("method_type"),
        mesh_type=params.get("mesh_type") or record.get("mesh_type") or "tensor",
        instance_type=instance_type, note=str(note)[:300], workspace_id=workspace_id,
        backend=backend, extra={"parent_job": job_id, "changes": changed or None}))


def vcpu_limit() -> int:
    """vCPUs the account may run at once (GEOINV3D_VCPU_LIMIT, aws.json "vcpu_limit")."""
    return int(os.environ.get("GEOINV3D_VCPU_LIMIT") or load_config().get("vcpu_limit", 36))


def vcpus_in_use() -> int:
    """vCPUs of the jobs that may still hold an instance."""
    total = 0
    for record in get_store().all():
        if record.get("status") in TERMINAL_STATUSES or record.get("status") == "STOPPED" \
                or record.get("backend") == "local":
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


def _label_run(run: dict, record: dict) -> None:
    """What the viewer shows of a job besides its result: its name, where it ran, and
    its id (the page re-runs it with changes from its node)."""
    run["_name"] = _run_label(record)
    run["_backend"] = record.get("backend", "ec2")
    run["_job_id"] = record["job_id"]
    run["_rerun"] = _has_inputs(record)


def _workflow_for(job_ids: list[str]) -> dict:
    """One DAG workflow holding the results of several finished jobs."""
    from ..viz.result_workflow import build_workflow, load_result

    runs = []
    result_dir = _upload_dir / "results"
    result_dir.mkdir(exist_ok=True)
    for job_id in job_ids:
        # an id the store does not know is refused before _refresh, which would keep a
        # record of it (an empty, "FAILED" one)
        record = get_store().get(job_id)
        if not record or not record.get("task_id"):
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
        run = load_result(backend_for(record).fetch_result(record, str(result_dir)))
        _label_run(run, record)
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


# ── Run-time estimates: rate × sensitivity entries × iterations, after a set-up time ──
# The defaults come from the Karnataka runs (5,040 data × 335,518 cells on c5.4xlarge: 30
# iterations in 8 min and 45 in 14 min, set-up included; joint runs of 10,081 data on
# c5.9xlarge in 22–28 min) and a run of 5,040 × 19,685 on a laptop; finished jobs refine them.
DEFAULT_RATE_S = {"local": 6e-9, "c5.xlarge": 3.2e-8, "c5.2xlarge": 1.6e-8, "c5.4xlarge": 8e-9,
                  "c5.9xlarge": 4e-9}
DEFAULT_SETUP_S = {"local": 5.0, "ec2": 90.0, "batch": 240.0}
JOINT_FACTOR = 2.0      # a joint iteration costs about two single ones per sensitivity entry
_NOT_PER_ITERATION = ("elastic_net_CDA", "group_lasso_ADMM")   # λ paths: not timed per iteration


def run_time_samples(records: list[dict]) -> dict:
    """Per machine ("local" or the instance type): the rates (s per sensitivity entry and
    iteration) of finished potential-field jobs; per backend: their set-up times (s)."""
    rates: dict[str, list] = {}
    setup: dict[str, list] = {}
    for r in records:
        s = r.get("summary") or {}
        n, m, it = s.get("n_data"), s.get("n_active_cells"), s.get("n_iterations")
        if r.get("status") != "SUCCEEDED" or not (n and m and it) or r.get("started") is None \
                or r.get("stopped") is None or s.get("regularization") in _NOT_PER_ITERATION \
                or r.get("method") not in ("gravity", "magnetics", "magnetic", "joint"):
            continue
        backend = r.get("backend") or "ec2"
        machine = "local" if backend == "local" else r.get("instance_type")
        run_s = (r["stopped"] - r["started"]) / 1000.0 - (DEFAULT_SETUP_S["local"] if backend == "local" else 0)
        work = float(n) * float(m) * float(it) * (JOINT_FACTOR if r.get("inversion_mode") == "joint" else 1)
        if run_s > 0 and machine:
            rates.setdefault(machine, []).append(run_s / work)
        if backend != "local" and r.get("created") is not None and r["started"] > r["created"]:
            setup.setdefault(backend, []).append((r["started"] - r["created"]) / 1000.0)
    return {"rates": rates, "setup": setup}


@app.get("/api/estimates")
async def estimates():
    """What the page needs to estimate a job's run time and cost: the rate of each machine
    (the median of its last 20 finished jobs, else a default) and the set-up time of each
    backend.  A run takes about setup + rate × data × cells × iterations (× joint_factor)."""
    import statistics
    samples = run_time_samples(get_store().all())
    rates = {k: {"rate": v, "n": 0} for k, v in DEFAULT_RATE_S.items()}
    for machine, values in samples["rates"].items():
        rates[machine] = {"rate": statistics.median(values[:20]), "n": len(values)}
    setup = {k: {"seconds": v, "n": 0} for k, v in DEFAULT_SETUP_S.items()}
    for backend, values in samples["setup"].items():
        setup[backend] = {"seconds": statistics.median(values[:20]), "n": len(values)}
    return JSONResponse({"ok": True, "rates": rates, "setup": setup, "joint_factor": JOINT_FACTOR})


@app.get("/api/inversion/{job_id}")
async def get_job_status(job_id: str):
    """One job: Batch status, the worker's progress, and the result summary."""
    return JSONResponse({"ok": True, **_refresh(job_id)})


@app.get("/api/inversion/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 100):
    """The last ``limit`` lines of the worker's log."""
    record = _refresh(job_id)
    try:
        lines = backend_for(record).tail_logs(record, limit=max(1, min(limit, 1000)))
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
        local_path = backend_for(record).fetch_result(record, str(result_dir))
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
        run = load_result(backend_for(record).fetch_result(record, str(result_dir)))
        _label_run(run, record)
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
    # the code that builds it is part of the key: a cache from before an update (e.g. the
    # 3D grids of octree jobs, once sampled from the top of the octree's box) is rebuilt
    stamp = f"{','.join(done)}|code {CODE_MTIME_AT_START:.0f}"
    key = hashlib.sha1(stamp.encode()).hexdigest()[:12] if done else ""
    counts = {"key": key, "finished": len(done), "running": sum(map(_running, jobs)),
              "running_local": sum(_running(j) and j.get("backend") == "local" for j in jobs),
              "jobs": len(jobs)}
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


# ── The size of a job's mesh: an OcTree keeps its finest cells only near the ground, so
# counting the core as if it were all fine cells overstates it many times ──
MESH_COUNT_MAX_FINE = 4_000_000     # fine cells in one layer above which no mesh is built


@lru_cache(maxsize=64)
def _mesh_cells(extent: tuple, h: float, dz: float, depth: float, pad: float, mesh_type: str,
                levels: tuple) -> dict:
    from ..cloud.worker import _build_octree_mesh, _build_tensor_mesh
    flat = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))   # noqa: E731 (a DEM adds a few layers)
    t0 = time.time()
    if mesh_type == "octree":
        mesh = _build_octree_mesh(extent, flat, h, dz, depth, pad, list(levels)).to_discretize()
    else:
        mesh = _build_tensor_mesh(extent, flat, False, h, dz, depth, pad).to_discretize()
    below = int((mesh.cell_centers[:, 2] < 0).sum())
    return {"n_cells": below, "n_total": int(mesh.n_cells), "seconds": round(time.time() - t0, 2)}


@app.post("/api/mesh/cells")
def mesh_cells(body: dict = Body(...)):
    """The cells below the ground of the mesh a job would build over ``extent`` [W, E, S, N]
    with core_cell_m, core_cell_z_m, depth_core_m, pad_distance_m and mesh_type (flat ground:
    a DEM adds a few layers), for the page's memory and time estimates.  ``too_large`` when
    the finest layer alone would hold more than MESH_COUNT_MAX_FINE cells."""
    try:
        ext = tuple(float(v) for v in body["extent"])
        h, dz = float(body["core_cell_m"]), float(body["core_cell_z_m"])
        depth, pad = float(body["depth_core_m"]), float(body.get("pad_distance_m") or 0.0)
        mesh_type = str(body.get("mesh_type") or "octree")
        levels = tuple(int(v) for v in body.get("octree_levels") or (4, 4, 4))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Bad mesh: {e}")
    if len(ext) != 4 or not (ext[1] > ext[0] and ext[3] > ext[2] and h > 0 and dz > 0 and depth > 0):
        raise HTTPException(status_code=400, detail="Bad mesh extent or cells")
    fine = ((ext[1] - ext[0] + 2 * pad) / h) * ((ext[3] - ext[2] + 2 * pad) / h)
    if fine > MESH_COUNT_MAX_FINE or (mesh_type != "octree" and fine * depth / dz > 3 * MESH_COUNT_MAX_FINE):
        return JSONResponse({"ok": True, "too_large": True, "n_cells": None})
    try:
        return JSONResponse({"ok": True, "too_large": False,
                             **_mesh_cells(ext, h, dz, depth, pad, mesh_type, levels)})
    except Exception as e:   # a mesh the worker could not build either
        raise HTTPException(status_code=400, detail=f"The mesh cannot be built: {e}")


# ── Ground and field for a survey area: a DEM, the IGRF, the coordinate system ──
def _check_crs(crs) -> str:
    from rasterio.crs import CRS
    text = str(crs or "").strip()
    if text.isdigit():
        text = f"EPSG:{text}"
    try:
        return CRS.from_user_input(text).to_string()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Unknown coordinate system '{crs}'")


@app.get("/api/crs")
async def check_crs(crs: str):
    """A coordinate system the page was given: its canonical name, and whether it is degrees."""
    from rasterio.crs import CRS
    name = _check_crs(crs)
    c = CRS.from_user_input(name)
    units = c.linear_units if not c.is_geographic else "degree"
    return JSONResponse({"ok": True, "crs": name, "geographic": bool(c.is_geographic),
                         "units": units, "description": c.to_wkt().split('"')[1] if '"' in c.to_wkt() else name})


@app.post("/api/dem")
def make_dem(body: dict = Body(...)):
    """A DEM over {"bounds": [west, east, south, north], "crs": ...} (the data's CRS), widened
    by "margin_m", from "source" ("auto": SRTM 30 m, else ETOPO 2022; "srtm"; "etopo").
    Downloads (and caches) the tiles it needs; GET /api/dem/{id} returns the GeoTIFF."""
    from ..io.dem import build_dem
    bounds, crs = body.get("bounds"), _check_crs(body.get("crs"))
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise HTTPException(status_code=400, detail="bounds must be [west, east, south, north]")
    try:
        meta = build_dem([float(v) for v in bounds], crs, source=str(body.get("source") or "auto"),
                         margin_m=float(body.get("margin_m") or 0.0),
                         max_pixels=int(body.get("max_pixels") or 3000))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=502, detail=f"Could not download the DEM tiles: {e}")
    return JSONResponse({"ok": True, **{k: v for k, v in meta.items() if k != "path"}})


@app.get("/api/dem/{dem_id}")
async def get_dem(dem_id: str):
    from ..io.dem import cached_dem
    meta = cached_dem(dem_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown DEM")
    return FileResponse(meta["path"], media_type="image/tiff", filename=meta["file_name"])


@app.get("/api/dem/{dem_id}/relief")
def dem_relief(dem_id: str, bounds: str):
    """The lowest and highest ground of a DEM made earlier within ``bounds`` (W,E,S,N in
    its CRS): the relief a mesh over that area follows, not that of the whole file."""
    import rasterio
    from rasterio.windows import from_bounds
    from ..io.dem import cached_dem
    meta = cached_dem(dem_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown DEM")
    try:
        w, e, s, n = (float(v) for v in bounds.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bounds: W,E,S,N")
    if not (e > w and n > s):
        raise HTTPException(status_code=400, detail="bounds: W,E,S,N")
    with rasterio.open(meta["path"]) as src:
        win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:      # outside the DEM
            return {"zmin": None, "zmax": None, "n": 0}
        z = src.read(1, window=win, masked=True).astype(float).filled(np.nan)
    z = z[np.isfinite(z)]
    if not z.size:
        return {"zmin": None, "zmax": None, "n": 0}
    return {"zmin": float(z.min()), "zmax": float(z.max()), "n": int(z.size)}


@app.get("/api/igrf")
async def igrf_field(x: float, y: float, crs: str = "EPSG:4326", alt_m: float = 0.0,
                     date: str = ""):
    """The IGRF-14 field at (x, y) of ``crs`` on ``date`` (YYYY-MM-DD, default today):
    F, I, D from true and from grid north, and ``inducing_field`` for a magnetic job."""
    import datetime as dt
    from ..methods.igrf import inducing_field
    crs = _check_crs(crs)
    try:
        f = inducing_field(x, y, crs, alt_m=alt_m, date=date or dt.date.today().isoformat())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "crs": crs, **{k: float(v) if hasattr(v, "dtype") else v
                                                     for k, v in f.items()}})


# ── Data enhancement: the maps an interpreter looks at before inverting (methods/enhance.py) ──
from collections import OrderedDict   # noqa: E402

_ENHANCE_DATA: "OrderedDict[str, tuple]" = OrderedDict()   # data id -> ((x, y, grid, note), method)
ENHANCE_KEEP = 4
ENHANCE_MAX_NODES_SIDE = 500    # larger grids are averaged down before the filters: the maps
                                # are a rough look (a few hundred pixels), not the processing
ENHANCE_MAPS = {   # key: (label, unit or None for the data's own, symmetric about 0)
    "data": ("Data", None, False), "rtp": ("Reduced to the pole", None, False),
    "continued": ("Continued upwards", None, False),
    "vdr": ("Vertical derivative", "/km", True), "thdr": ("Total horizontal derivative", "/km", False),
    "asa": ("Analytic signal amplitude", "/km", False), "tilt": ("Tilt angle", "°", True),
    "theta": ("Theta map (THDR / ASA)", "", False), "tdr_thdr": ("Horizontal derivative of the tilt", "°/km", False),
    "nstd": ("Normalized standard deviation", "", False),
    "nvdr_thdr": ("NVDR of THDR", "", False), "nvdr_tilt": ("NVDR of the tilt", "", False),
    "nvdr_theta": ("NVDR of theta", "", False), "nvdr_nstd": ("NVDR of NSTD", "", False),
    "regional": ("Regional field", None, False), "residual": ("Residual field", None, True),
}


def enhancement_grid(xg, yg, g):
    """The grid the maps are computed on, and a note: a survey grid of millions of nodes is
    averaged down first (the filters would need many GB; the maps are a few hundred pixels)."""
    from ..methods import enhance as E
    k = int(np.ceil(max(g.shape) / ENHANCE_MAX_NODES_SIDE))
    if k <= 1:
        return xg, yg, g, None
    d = float(xg[1] - xg[0])
    note = (f"Computed on the grid averaged {k} × {k} ({d * k:g} m) from its {g.shape[1]} × "
            f"{g.shape[0]} nodes of {d:g} m, for the maps")
    xs, ys, c = E.coarsen(xg, yg, g, k)
    return xs, ys, c, note


def enhancement_maps(grid, method: str, p: dict) -> dict:
    """The requested maps (``p["products"]``) of a gridded dataset (xg, yg, g, note), for the page."""
    from ..methods import enhance as E
    xg, yg, g, note = grid
    notes, want = ([note] if note else []), set(p.get("products") or ["data"])
    dx, dy = float(xg[1] - xg[0]), float(yg[1] - yg[0])
    maps = {"data": g}
    magnetic = method in ("magnetics", "magnetic")
    work = g
    if magnetic and p.get("inc") is not None:
        inc, dec = float(p["inc"]), float(p.get("dec") or 0.0)
        maps["rtp"] = E.rtp(g, dx, dy, inc, dec)
        work = maps["rtp"]
        if abs(inc) < 30:
            notes.append(f"Near the magnetic equator (I = {inc:g}°) the reduction to the pole is "
                         "damped along the declination; trust the derivatives of the RTP less there")
    uc = float(p.get("uc_m") or 0.0)
    if uc > 0:
        work = maps["continued"] = E.upward(work, dx, dy, uc)
    if want & {"vdr", "thdr", "asa", "tilt", "theta", "tdr_thdr", "nstd"} or any(k.startswith("nvdr") for k in want):
        maps.update(E.edges(work, dx, dy, int(p.get("window") or 5)))
        for k in ("thdr", "tilt", "theta", "nstd"):
            if f"nvdr_{k}" in want:
                maps[f"nvdr_{k}"] = E.nvdr(maps[k], dx, dy)
        for k in ("vdr", "thdr", "asa"):
            maps[k] = maps[k] * 1000.0          # per km
    reg = p.get("regional")
    if reg and reg.get("method") not in (None, "none") and want & {"regional", "residual"}:
        from ..methods.regional import remove_regional
        # on the data the inversion fits: as measured, or reduced to the pole
        base = maps["rtp"] if p.get("invert_rtp") and "rtp" in maps else g
        X, Y = np.meshgrid(xg, yg)
        ok = np.isfinite(base)
        res, info = remove_regional(np.c_[X[ok], Y[ok]], base[ok], reg)
        maps["residual"] = np.full_like(g, np.nan); maps["residual"][ok] = res
        maps["regional"] = base - maps["residual"]
        notes.append(f"Regional field: {info['label']} (on the data "
                     f"{'reduced to the pole' if base is not g else 'as measured'}, as inverted); "
                     f"std {info['data_std_before']:.4g} → {info['data_std_after']:.4g}")
        if reg.get("method") == "upward":
            notes.append("Upward continuation also weakens the broad field itself, so the residual "
                         "keeps part of it; a Butterworth low-pass separates it more cleanly")
    step = max(1, int(np.ceil(max(g.shape) / int(p.get("max_px") or 300))))
    out = {}
    for k in want:
        if k not in maps:
            continue
        a = maps[k][::step, ::step]
        label, unit, sym = ENHANCE_MAPS.get(k, (k, "", False))
        finite = a[np.isfinite(a)]
        out[k] = {"label": label, "unit": unit, "symmetric": sym,
                  "range": [float(np.percentile(finite, 1)), float(np.percentile(finite, 99))] if finite.size else None,
                  "values": [[None if not np.isfinite(val) else float(f"{val:.4g}") for val in row] for row in a]}
    return {"x": [float(v) for v in xg[::step]], "y": [float(v) for v in yg[::step]],
            "spacing_m": dx, "shape": list(g.shape), "display_step": step, "maps": out, "notes": notes}


@app.post("/api/enhance")
async def enhance_data(files: list[UploadFile] = File(default=[]), data_id: str = Form(""),
                       method: str = Form("magnetics"), component: str = Form(""),
                       crs: str = Form(""), params_json: str = Form("{}")):
    """Maps of a dataset for interpretation (methods/enhance.py).  The first call sends the
    files (read as the worker reads them, gridded at their spacing) and gets a ``data_id``;
    later calls send only the id with other settings.  ``params_json``: products (keys of
    ENHANCE_MAPS), inc and dec (the field, declination from grid north, for the reduction to
    the pole), uc_m (continue upwards first), window (NSTD), regional (a methods/regional
    spec), invert_rtp (the regional and residual of the data reduced to the pole, when the
    job inverts those), max_px (the returned maps' size)."""
    from starlette.concurrency import run_in_threadpool
    try:
        p = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"params_json is not JSON: {e}")
    if files:
        from ..cloud.worker import _DEFAULT_COMPONENT, _GRID_FORMATS, _read_observations, grid_window
        from ..io.readers import detect_format
        from ..methods.enhance import to_grid
        folder = Path(tempfile.mkdtemp(prefix="enhance_", dir=_upload_dir))
        digest = hashlib.sha1((method + component + crs).encode())

        def load(paths):
            # one grid file stays a grid (a survey grid has tens of millions of nodes); station
            # tables (or several files) are gridded at their spacing
            if len(paths) == 1 and detect_format(str(paths[0])) in _GRID_FORMATS:
                return enhancement_grid(*grid_window(str(paths[0]), crs or None))
            xs, ys, vs = [], [], []
            for path in paths:
                locs, values, _ = _read_observations(
                    str(path), component or _DEFAULT_COMPONENT.get(method, "tmi"), 1, None, None, crs or None)
                xs.append(locs[:, 0]); ys.append(locs[:, 1]); vs.append(values)
            return enhancement_grid(*to_grid(np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)))

        try:
            paths = []
            for f in files:
                path = folder / Path(f.filename or "data").name
                with open(path, "wb") as out:      # in pieces: a survey grid is hundreds of MB
                    while chunk := await f.read(1 << 22):
                        digest.update(chunk)
                        out.write(chunk)
                paths.append(path)
            grid = await run_in_threadpool(load, paths)
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        data_id = digest.hexdigest()[:12]
        _ENHANCE_DATA[data_id] = (grid, method)
        while len(_ENHANCE_DATA) > ENHANCE_KEEP:
            _ENHANCE_DATA.popitem(last=False)
    if data_id not in _ENHANCE_DATA:
        raise HTTPException(status_code=404, detail="Unknown data: send the files again")
    _ENHANCE_DATA.move_to_end(data_id)
    grid, m = _ENHANCE_DATA[data_id]
    try:
        out = await run_in_threadpool(enhancement_maps, grid, m, p)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True, "data_id": data_id, **out})


# Map layers: points (mines, towns) and lines or outlines (geology) drawn over every run
# of the workspace, as the page made them from GeoJSON or CSV files:
#   {"name", "color", "kind": "points" | "lines", "items": [...]}, items being
#   {"x", "y", "label"} (points) or {"xy": [[x, y], ...], "label", "closed"} (lines)
MAX_LAYERS_BYTES = 5_000_000


def check_layers(layers) -> list:
    if not isinstance(layers, list):
        raise HTTPException(status_code=400, detail="layers must be a list")
    for layer in layers:
        if not isinstance(layer, dict) or not isinstance(layer.get("items"), list) \
                or layer.get("kind") not in ("points", "lines"):
            raise HTTPException(status_code=400, detail=(
                "Each layer needs a kind ('points' or 'lines') and a list of items"))
    if len(json.dumps(layers)) > MAX_LAYERS_BYTES:
        raise HTTPException(status_code=413, detail=(
            f"The layers are larger than {MAX_LAYERS_BYTES // 1_000_000} MB: simplify the outlines"))
    return layers


@app.get("/api/workspaces/{workspace_id}/layers")
async def workspace_layers(workspace_id: str):
    return JSONResponse({"ok": True, "layers": _read_workspace(workspace_id).get("layers") or []})


@app.put("/api/workspaces/{workspace_id}/layers")
async def set_workspace_layers(workspace_id: str, body: dict = Body(...)):
    """Replace the workspace's map layers."""
    ws = _read_workspace(workspace_id)
    ws["layers"] = check_layers(body.get("layers"))
    _write_json(_workspace_path(workspace_id), ws)
    return JSONResponse({"ok": True, "layers": ws["layers"]})


@app.delete("/api/inversion/{job_id}")
async def cancel_job(job_id: str):
    """Stop a job: a queued job is cancelled, a running one is terminated."""
    record = get_store().get(job_id) or {"job_id": job_id}
    try:
        backend_for(record).cancel(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    get_store().update(job_id, cancel_requested=True)
    return JSONResponse({"ok": True, "message": f"Job {job_id} stopped", **_refresh(job_id)})


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _job_files(record: dict) -> list[Path]:
    """The folders and files this server keeps for a job: its inputs (for re-runs), the
    local run's folder, the result fetched from AWS and the temporary copies."""
    task = str(record.get("task_id") or "")
    if not _SAFE_NAME.match(task) or ".." in task:     # never a path outside the folders
        return []
    paths = [_inputs_dir(task)]
    backend = backend_for(record)
    if record.get("backend") == "local" and hasattr(backend, "job_dir"):
        paths.append(backend.job_dir(record))
    elif hasattr(backend, "result_dir"):
        paths.append(backend.result_dir(record))
    paths += sorted((_upload_dir / "results").glob(f"{task}*"))
    return [p for p in paths if p.exists()]


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job that has ended, and everything this server keeps for it: its record,
    its result, its inputs, the local run's folder, its place in the workspaces.  A job
    still queued or running has to be stopped first (DELETE /api/inversion/{id}).  Saved
    comparisons hold their own copy of the results and keep it; nothing on AWS is touched
    (an EC2 job's instance ends with the job)."""
    store = get_store()
    if not store.get(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    record = _refresh(job_id)
    if _running(record):
        raise HTTPException(status_code=409, detail="The job is still queued or running: stop it first")
    removed = []
    for path in _job_files(record):
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(str(path))
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not delete {path}: {e}")
    store.delete(job_id)
    workspaces = []
    for path in _workspaces_dir().glob("*.json"):
        try:
            ws = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job_id in ws.get("job_ids", []):
            ws["job_ids"] = [j for j in ws["job_ids"] if j != job_id]
            _write_json(path, ws)
            workspaces.append(ws.get("id"))
    return JSONResponse({"ok": True, "job_id": job_id, "removed": removed, "workspaces": workspaces})


# ── Slices of a result at its mesh's resolution (viz/sections.py), for the 3D tab ──
_MODELS: "OrderedDict[str, object]" = OrderedDict()
MODELS_KEEP = 6


def _result_model(job_id: str):
    """The ResultModel of a finished job, the last MODELS_KEEP kept in memory."""
    from ..viz.result_workflow import load_result
    from ..viz.sections import ResultModel
    if job_id in _MODELS:
        _MODELS.move_to_end(job_id)
        return _MODELS[job_id]
    record = get_store().get(job_id)
    if not record or not record.get("task_id"):
        raise HTTPException(status_code=404, detail="Unknown job")
    if _refresh(job_id).get("display_status") != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="The job has no result yet")
    result_dir = _upload_dir / "results"
    result_dir.mkdir(exist_ok=True)
    model = ResultModel(load_result(backend_for(record).fetch_result(record, str(result_dir))))
    _MODELS[job_id] = model
    while len(_MODELS) > MODELS_KEEP:
        _MODELS.popitem(last=False)
    return model


@app.get("/api/inversion/{job_id}/section")
def job_section(job_id: str, kind: str = "line", ref: str = "ground",
                x0: float = None, y0: float = None, x1: float = None, y1: float = None,
                level: float = None, depth_max: float = None, field: str = "model"):
    """A slice of a job's model sampled from its own mesh: ``kind`` "line" (A x0, y0 to
    B x1, y1; ``ref`` "ground": depth below the ground, "elev": elevation), "plan" (``level``
    metres below the ground, or an elevation with ``ref`` "elev") or "strike" (the default
    profile across the strike of the strongest bodies)."""
    if ref not in ("ground", "elev"):
        raise HTTPException(status_code=400, detail="ref: ground or elev")
    model = _result_model(job_id)
    try:
        if field == "agreement":   # the job's robustness runs: where they agree
            from ..viz.sections import agreement_slice
            _, _, models = _robust_models(job_id)
            if len(models) < 2:
                raise HTTPException(status_code=409, detail="Fewer than two robustness runs have finished")
            if kind == "plan":
                return JSONResponse(agreement_slice(model, models, "plan", level=level, ref=ref))
            return JSONResponse(agreement_slice(model, models, "line", x0=x0, y0=y0, x1=x1, y1=y1,
                                                ref=ref, depth_max=depth_max))
        if kind == "strike":
            return JSONResponse(model.strike_profile(level))
        if kind == "plan":
            if level is None:
                raise ValueError("a plan slice needs its level")
            return JSONResponse(model.plan(level, ref))
        if kind == "line":
            if None in (x0, y0, x1, y1):
                raise ValueError("a line needs x0, y0, x1, y1")
            return JSONResponse(model.line(x0, y0, x1, y1, ref, depth_max))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail="kind: line, plan or strike")


# ── Robustness: the same data inverted with several standard settings, and where the
# models agree (viz/sections.py: agreement, ensemble_summary) ──
# The regularizations users choose between, and depth weightings weaker and stronger than
# the default: what the data cannot decide comes out as disagreement.
ROBUSTNESS_VARIANTS = [
    ("L1–L2, α 0.8", {"regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "irls",
                      "depth_weighting": "sensitivity"}),
    ("Lp 0,1,1,1", {"regularization_type": "sparse", "norms": [0, 1, 1, 1], "depth_weighting": "sensitivity"}),
    ("Lp 0,2,2,2", {"regularization_type": "sparse", "norms": [0, 2, 2, 2], "depth_weighting": "sensitivity"}),
    ("smooth L2", {"regularization_type": "l2", "depth_weighting": "sensitivity"}),
    ("smooth L2, depth weighting β 1.5", {"regularization_type": "l2", "depth_weighting": "depth",
                                          "depth_weighting_exponent": 1.5}),
    ("smooth L2, depth weighting β 3", {"regularization_type": "l2", "depth_weighting": "depth",
                                        "depth_weighting_exponent": 3.0}),
]
ROBUSTNESS_MAX_DATA = 6000      # above this, the variants thin the data to twice their spacing


def _robustness_members(job_id: str) -> list[dict]:
    return [_refresh(j["job_id"]) for j in get_store().all() if j.get("robustness_of") == job_id]


@app.post("/api/inversion/{job_id}/robustness")
async def start_robustness(job_id: str, body: dict = Body(default={})):
    """Invert a finished job's data again with ROBUSTNESS_VARIANTS (one job each, a group),
    all with automatic errors and, for more than ROBUSTNESS_MAX_DATA data, thinned to twice
    their spacing, on this computer unless ``backend`` says "aws"."""
    record, folder, base, _ = _job_inputs(job_id)
    if _refresh(job_id).get("display_status") != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="The job has not finished")
    if _robustness_members(job_id):
        raise HTTPException(status_code=409, detail="A robustness check of this job exists already")
    base = {k: v for k, v in base.items() if k not in ("task_id", "group", "preview_of")}
    if base.get("inversion_mode") == "joint" or len(base.get("datasets") or []) != 1:
        raise HTTPException(status_code=400, detail="The robustness check is for single gravity or magnetic inversions")
    ds = dict(base["datasets"][0])
    if ds.get("type") not in ("gravity", "magnetic"):
        raise HTTPException(status_code=400, detail="The robustness check is for gravity or magnetic data")
    mvi = (ds.get("method_kwargs") or {}).get("magnetization") == "vector"
    summary = record.get("summary") or {}
    ds_change = {"noise_pct": "auto", "noise_floor": "auto"}
    if not ds.get("decimate_spacing_m") and (summary.get("n_data") or 0) > ROBUSTNESS_MAX_DATA:
        spacing = [d.get("spacing_m") for d in ((_result_model(job_id).meta.get("mesh_design") or {})
                                                 .get("data_spacing") or []) if d.get("spacing_m")]
        if spacing:
            ds_change["decimate_spacing_m"] = round(2 * min(spacing), 3)
    backend = body.get("backend") or "local"
    instance_type = body.get("instance_type") or "c5.2xlarge"
    label = f"Robustness of {record.get('note') or job_id}"[:200]
    started = []
    for name, changes in ROBUSTNESS_VARIANTS:
        if mvi and changes["regularization_type"] == "l1l2":
            continue                               # MVI takes Lp norms or L2
        ch = {**changes, "param_mode": "manual", "beta_selection": "auto",
              "datasets": [ds_change], "group": {"id": f"rob-{record['task_id']}", "label": label,
                                                 "variant": name}}
        params = merge_params(base, {k: v for k, v in ch.items() if k != "group"})
        params["group"] = ch["group"]
        started.append(_launch(
            uuid.uuid4().hex[:12], sorted((folder / "data").iterdir()), params,
            method=record.get("method") or params.get("method_type"),
            mesh_type=params.get("mesh_type") or record.get("mesh_type") or "tensor",
            instance_type=instance_type, note=f"{name} · robustness of {record.get('note') or job_id}"[:300],
            workspace_id=record.get("workspace_id") or "", backend=backend,
            extra={"parent_job": job_id, "robustness_of": job_id, "changes": changes}))
    return JSONResponse({"ok": True, "jobs": [j["job_id"] for j in started], "group_label": label,
                         "thinned_to_m": ds_change.get("decimate_spacing_m")})


_ROBUST_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _robust_models(job_id: str):
    members = _robustness_members(job_id)
    done = [m for m in members if m.get("display_status") == "SUCCEEDED"]
    return members, done, [_result_model(m["job_id"]) for m in done]


@app.get("/api/inversion/{job_id}/robustness")
def robustness(job_id: str, grid: bool = False):
    """A job's robustness check: its runs and, once two have finished, where their models
    agree (``summary``), and with ``grid`` the agreement on the job's 3D grid."""
    if not get_store().get(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    members, done, models = _robust_models(job_id)
    out = {"job_id": job_id, "n_runs": len(members), "n_done": len(done),
           "runs": [{"job_id": m["job_id"], "variant": m.get("variant"), "status": m.get("display_status"),
                     "chi2_per_datum": ((m.get("summary") or {}).get("convergence") or {}).get("chi2_per_datum")}
                    for m in members]}
    if len(models) >= 2:
        from ..viz.sections import agreement_grid, ensemble_summary
        key = job_id + "|" + ",".join(m["job_id"] for m in done) + ("|grid" if grid else "")
        if key not in _ROBUST_CACHE:
            base = _result_model(job_id)
            res = {"summary": ensemble_summary(base, models)}
            res["summary"]["variants"] = [m.get("variant") for m in done]
            if grid:
                res["grid"] = agreement_grid(base.meta, models)
            _ROBUST_CACHE[key] = res
            while len(_ROBUST_CACHE) > 8:
                _ROBUST_CACHE.popitem(last=False)
        out.update(_ROBUST_CACHE[key])
    return JSONResponse(out)


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
        backend_for(record).request_finish(record)
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
        backend_for(record).fetch(record)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, **_refresh(job_id)})


PAGE_PATH = Path(__file__).resolve().parent.parent / "viz" / "dag_interactive.html"
CODE_DIR = Path(__file__).resolve().parent.parent


def _code_mtime() -> float:
    """The newest modification time of the package's Python files."""
    return max((p.stat().st_mtime for p in CODE_DIR.rglob("*.py")), default=0.0)


# Python is loaded once: code changed after this server started is not in use until it
# restarts (the page itself is read on every request, so it is always the new one)
CODE_MTIME_AT_START = _code_mtime()


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
        # where jobs can run: this machine (its cores and memory) and AWS
        "local": get_local_backend().resources(),
        "comparisons_dir": str(_comparisons_dir()),
        "code_changed": _code_mtime() > CODE_MTIME_AT_START + 1e-3,
    })


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GeoInv3D API Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ec2", "batch"], default=None,
                        help="AWS backend (or set GEOINV3D_BACKEND; default ec2); jobs can "
                             "also run on this machine, chosen on the page")
    parser.add_argument("--local-jobs", type=int, default=None,
                        help="Jobs run at once on this machine (default 1; each uses every core)")
    parser.add_argument("--region", default=None,
                        help="AWS region (or set GEOINV3D_AWS_REGION)")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.backend:
        os.environ["GEOINV3D_BACKEND"] = args.backend
    if args.region:
        os.environ["GEOINV3D_AWS_REGION"] = args.region
    if args.local_jobs:
        os.environ["GEOINV3D_LOCAL_JOBS"] = str(args.local_jobs)

    cfg = backend_settings()
    print(f"Starting GeoInv3D API on {args.host}:{args.port}")
    local = get_local_backend().resources()
    print(f"  Local:     {local['cpus']} cores, {local['memory_gb']} GB, "
          f"{local['max_jobs']} job(s) at a time")
    print(f"  AWS:       {cfg['backend']} in {cfg['region']}")
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
