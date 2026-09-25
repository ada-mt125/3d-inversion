"""AWS integration: S3 upload/download, Batch job management, and
end-to-end inversion pipeline.

Provides AWSRunner for two workflows:

  Workflow 1 — Pre-packed task (existing):
    1. pack_task() creates a .zip with mesh, data, and parameters
    2. AWSRunner.submit() uploads to S3 and submits an AWS Batch job
    3. AWSRunner.wait() checks job status
    4. AWSRunner.fetch_result() downloads the result archive

  Workflow 2 — Raw data pipeline (new):
    1. AWSRunner.upload_data() uploads raw survey files (GeoTIFF, CSV, etc.)
    2. AWSRunner.submit_pipeline() sends inversion params + data location
    3. Worker loads data, builds mesh, runs inversion
    4. AWSRunner.fetch_result() downloads results

Prerequisites:
    - boto3 installed
    - AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)
    - S3 bucket and Batch job queue/definition created (see deploy/ folder)
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .task import InversionTask, pack_task, unpack_task

# vCPUs and memory (MiB) requested for each instance choice on the upload page.
# Batch places the job on an instance of its compute environment that fits the
# request; memory stays below the instance's RAM to leave room for the agent.
INSTANCE_RESOURCES = {
    "c5.xlarge": (4, 7000),
    "c5.2xlarge": (8, 14500),
    "c5.4xlarge": (16, 29500),
    "c5.9xlarge": (36, 68000),
}
TERMINAL_STATUSES = ("SUCCEEDED", "FAILED")
LOG_GROUP = "/aws/batch/geoinv3d"   # deploy/batch-job-definition.json


def task_id_from_job_name(name: str) -> str | None:
    """Task id encoded in the Batch job name (geoinv3d[-pipeline]-<task_id>)."""
    for prefix in ("geoinv3d-pipeline-", "geoinv3d-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


class AWSRunner:
    """Submit and manage inversion jobs on AWS."""

    def __init__(
        self,
        bucket: str,
        region: str = "eu-west-1",
        job_queue: str = "geoinv3d-queue",
        job_definition: str = "geoinv3d-worker",
        prefix: str = "geoinv3d/jobs",
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.job_queue = job_queue
        self.job_definition = job_definition
        self.prefix = prefix
        self._s3 = None
        self._batch = None
        self._logs = None

    @property
    def s3(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    @property
    def batch(self):
        if self._batch is None:
            import boto3
            self._batch = boto3.client("batch", region_name=self.region)
        return self._batch

    @property
    def logs(self):
        if self._logs is None:
            import boto3
            self._logs = boto3.client("logs", region_name=self.region)
        return self._logs

    # ---- Workflow 1: Pre-packed task ----

    def submit(
        self,
        task: InversionTask,
        local_archive: Optional[str] = None,
    ) -> str:
        """Pack, upload, and submit a task.  Returns the Batch job ID."""
        if not task.task_id:
            task.task_id = uuid.uuid4().hex[:12]

        if local_archive is None:
            local_archive = f"{task.task_id}.geoinv3d.zip"

        pack_task(task, local_archive)

        s3_key = f"{self.prefix}/{task.task_id}/input.zip"
        print(f"[AWS] Uploading {local_archive} -> s3://{self.bucket}/{s3_key}")
        self.s3.upload_file(local_archive, self.bucket, s3_key)

        response = self.batch.submit_job(
            jobName=f"geoinv3d-{task.task_id}",
            jobQueue=self.job_queue,
            jobDefinition=self.job_definition,
            containerOverrides={
                "environment": [
                    {"name": "TASK_BUCKET", "value": self.bucket},
                    {"name": "TASK_KEY", "value": s3_key},
                    {"name": "TASK_ID", "value": task.task_id},
                    {"name": "RESULT_PREFIX", "value": f"{self.prefix}/{task.task_id}"},
                    {"name": "PIPELINE_MODE", "value": "task"},
                ],
            },
        )

        job_id = response["jobId"]
        print(f"[AWS] Submitted job: {job_id}")
        return job_id

    # ---- Workflow 2: Raw data pipeline ----

    def upload_data(
        self,
        local_paths: list[str],
        task_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Upload raw data files to S3.

        Args:
            local_paths: Local file paths to upload (GeoTIFF, CSV, etc.).
            task_id: Optional ID; generated if not provided.

        Returns:
            (task_id, data_prefix) — the S3 prefix where files were uploaded.
        """
        if task_id is None:
            task_id = uuid.uuid4().hex[:12]

        data_prefix = f"{self.prefix}/{task_id}/data"
        for local_path in local_paths:
            fname = Path(local_path).name
            s3_key = f"{data_prefix}/{fname}"
            print(f"[AWS] Uploading {local_path} -> s3://{self.bucket}/{s3_key}")
            self.s3.upload_file(str(local_path), self.bucket, s3_key)

        print(f"[AWS] Uploaded {len(local_paths)} file(s) to s3://{self.bucket}/{data_prefix}/")
        return task_id, data_prefix

    def submit_pipeline(
        self,
        task_id: str,
        data_prefix: str,
        params: dict,
        instance_type: Optional[str] = None,
    ) -> str:
        """Submit a raw-data pipeline job.

        The worker will download data files from S3, build the mesh,
        and run the inversion according to params.

        Args:
            task_id: Task identifier (from upload_data).
            data_prefix: S3 prefix where data files live.
            instance_type: One of INSTANCE_RESOURCES; requests that many vCPUs
                and that much memory (else the job definition's defaults).
            params: Inversion parameters dict. Keys:
                method_type: "gravity" or "magnetics"
                regularization_type: "smooth" or "sparse" (default "sparse")
                data_file: filename of the primary data file (auto-detected if omitted)
                aoi: [west, east, south, north] (optional)
                core_cell_m: horizontal cell size (default 500)
                core_cell_z_m: vertical cell size (default 250)
                depth_core_m: depth of fine mesh (default 3000)
                pad_distance_m: padding distance (default 2000)
                decimate_stride: data thinning (default 1)
                norms: [s, x, y, z] regularization norms (default [0, 2, 2, 1])
                max_iter: max outer iterations (default 25)
                irls_cooling_factor: IRLS cooling (default 1.1)
                max_irls_iterations: max IRLS iterations (default 12)
                use_preconditioner: enable CG preconditioner (default true)
                noise_pct: relative noise (default 0.05)
                noise_floor: absolute noise floor (default 0.5)
                bounds_lower, bounds_upper: model bounds (optional)
                method_kwargs: extra method constructor args (optional)

        Returns:
            AWS Batch job ID.
        """
        params["task_id"] = task_id
        params_key = f"{self.prefix}/{task_id}/params.json"
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(params, f, indent=2)
            params_path = f.name
        try:
            self.s3.upload_file(params_path, self.bucket, params_key)
        finally:
            os.unlink(params_path)

        overrides = {
            "environment": [
                {"name": "TASK_BUCKET", "value": self.bucket},
                {"name": "TASK_ID", "value": task_id},
                {"name": "RESULT_PREFIX", "value": f"{self.prefix}/{task_id}"},
                {"name": "PIPELINE_MODE", "value": "data"},
                {"name": "PIPELINE_PARAMS", "value": params_key},
                {"name": "DATA_PREFIX", "value": data_prefix},
            ],
        }
        if instance_type is not None:
            if instance_type not in INSTANCE_RESOURCES:
                raise ValueError(f"Unknown instance type '{instance_type}' "
                                 f"(expected one of {sorted(INSTANCE_RESOURCES)})")
            vcpus, memory = INSTANCE_RESOURCES[instance_type]
            overrides["resourceRequirements"] = [
                {"type": "VCPU", "value": str(vcpus)},
                {"type": "MEMORY", "value": str(memory)},
            ]
        response = self.batch.submit_job(
            jobName=f"geoinv3d-pipeline-{task_id}",
            jobQueue=self.job_queue,
            jobDefinition=self.job_definition,
            containerOverrides=overrides,
        )

        job_id = response["jobId"]
        print(f"[AWS] Submitted pipeline job: {job_id}")
        return job_id

    def run_pipeline(
        self,
        local_paths: list[str],
        params: dict,
        poll_interval: int = 30,
        timeout: int = 7200,
    ) -> dict:
        """End-to-end: upload data, run inversion on AWS, download results.

        Args:
            local_paths: Local data files to upload.
            params: Inversion parameters (see submit_pipeline).
            poll_interval: Seconds between status checks.
            timeout: Max seconds to wait.

        Returns:
            Status dict with 'result_path' on success.
        """
        task_id, data_prefix = self.upload_data(local_paths)
        job_id = self.submit_pipeline(task_id, data_prefix, params)
        status = self.wait(job_id, poll_interval, timeout)
        if status["status"] == "SUCCEEDED":
            result_path = self.fetch_result(task_id)
            status["result_path"] = result_path
        return status

    # ---- Common operations ----

    def poll(self, job_id: str) -> dict:
        """Check job status.

        Returns {status, name, task_id, created, started, stopped (ms since
        the epoch), reason, log_stream, exit_code}; missing ones are omitted.
        Batch statuses: SUBMITTED, PENDING, RUNNABLE (waiting for an
        instance), STARTING (pulling the image), RUNNING, SUCCEEDED, FAILED
        (also after a cancel or terminate).
        """
        resp = self.batch.describe_jobs(jobs=[job_id])
        if not resp["jobs"]:
            return {"status": "UNKNOWN"}

        job = resp["jobs"][0]
        result = {
            "status": job["status"],
            "name": job.get("jobName", ""),
        }
        task_id = task_id_from_job_name(result["name"])
        if task_id:
            result["task_id"] = task_id
        for key, name in (("createdAt", "created"), ("startedAt", "started"),
                          ("stoppedAt", "stopped")):
            if key in job:
                result[name] = job[key]
        if job.get("statusReason"):
            result["reason"] = job["statusReason"]
        container = job.get("container") or {}
        if container.get("logStreamName"):
            result["log_stream"] = container["logStreamName"]
        if container.get("exitCode") is not None:
            result["exit_code"] = container["exitCode"]
        return result

    def wait(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> dict:
        """Block until the job finishes.  Returns final status."""
        terminal = {"SUCCEEDED", "FAILED"}
        elapsed = 0
        while elapsed < timeout:
            status = self.poll(job_id)
            print(f"[AWS] Job {job_id[:12]}... status={status['status']} "
                  f"({elapsed}s elapsed)")
            if status["status"] in terminal:
                return status
            time.sleep(poll_interval)
            elapsed += poll_interval

        return {"status": "TIMEOUT", "elapsed": elapsed}

    def fetch_result(
        self,
        task_id: str,
        output_dir: str = ".",
    ) -> str:
        """Download the result archive from S3.  Returns local path."""
        s3_key = f"{self.prefix}/{task_id}/result.zip"
        local_path = str(Path(output_dir) / f"{task_id}_result.zip")

        print(f"[AWS] Downloading s3://{self.bucket}/{s3_key}")
        self.s3.download_file(self.bucket, s3_key, local_path)
        print(f"[AWS] Saved to {local_path}")
        return local_path

    def submit_and_wait(
        self,
        task: InversionTask,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> dict:
        """Submit a task and block until completion."""
        job_id = self.submit(task)
        status = self.wait(job_id, poll_interval, timeout)
        if status["status"] == "SUCCEEDED":
            result_path = self.fetch_result(task.task_id)
            status["result_path"] = result_path
        return status

    def list_jobs(self, status_filter: str = "RUNNING") -> list[dict]:
        """List jobs in the queue with the given status."""
        resp = self.batch.list_jobs(
            jobQueue=self.job_queue,
            jobStatus=status_filter,
        )
        return [
            {
                "job_id": j["jobId"],
                "name": j["jobName"],
                "status": j["status"],
                "created": j.get("createdAt", 0),
            }
            for j in resp.get("jobSummaryList", [])
        ]

    def cancel(self, job_id: str, reason: str = "Cancelled by user") -> None:
        """Stop a job in any state: queued jobs are cancelled, running ones killed.

        Uses TerminateJob: CancelJob only affects jobs that have not reached
        STARTING, and for a running job it succeeds without stopping it, so
        the instance would keep running (and billing).  The job then ends as
        FAILED with ``reason`` as its status reason.
        """
        self.batch.terminate_job(jobId=job_id, reason=reason)
        print(f"[AWS] Terminated job {job_id}")

    def _read_json(self, key: str) -> Optional[dict]:
        """A small JSON object from S3, or None if it does not exist (yet)."""
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return json.loads(body)

    def progress(self, task_id: str) -> Optional[dict]:
        """The worker's progress report (see worker.S3Progress), or None."""
        return self._read_json(f"{self.prefix}/{task_id}/progress.json")

    def result_summary(self, task_id: str) -> Optional[dict]:
        """result.json of a finished job (no model arrays), or None."""
        return self._read_json(f"{self.prefix}/{task_id}/result.json")

    def tail_logs(self, log_stream: str, limit: int = 100,
                  log_group: str = LOG_GROUP) -> list[str]:
        """The last ``limit`` lines the worker printed (CloudWatch Logs)."""
        resp = self.logs.get_log_events(
            logGroupName=log_group, logStreamName=log_stream,
            limit=limit, startFromHead=False,
        )
        return [e["message"] for e in resp.get("events", [])]
