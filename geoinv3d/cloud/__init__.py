"""Cloud computation: package, submit, and retrieve inversion jobs on AWS.

Two workflows:
  1. Pre-packed task: pack_task -> AWSRunner.submit -> fetch_result
  2. Raw data pipeline: AWSRunner.upload_data -> submit_pipeline -> fetch_result
"""

from .task import InversionTask, pack_task, unpack_task
from .aws import AWSRunner
from .worker import execute_task, pack_result, run_data_pipeline

__all__ = [
    "InversionTask", "pack_task", "unpack_task",
    "AWSRunner",
    "execute_task", "pack_result", "run_data_pipeline",
]
