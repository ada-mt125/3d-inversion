"""Cloud computation: package, submit, and retrieve inversion jobs on AWS.

Two workflows:
  1. Pre-packed task: pack_task -> AWSRunner.submit -> fetch_result
  2. Raw data pipeline: AWSRunner.upload_data -> submit_pipeline -> fetch_result
"""

from .task import InversionTask, pack_task, unpack_task
from .aws import AWSRunner

__all__ = [
    "InversionTask", "pack_task", "unpack_task",
    "AWSRunner",
    "execute_task", "pack_result", "run_data_pipeline",
]

_FROM_WORKER = {"execute_task", "pack_result", "run_data_pipeline"}


def __getattr__(name):
    # The worker is imported only when one of its functions is asked for: the instances
    # run it as ``python -m geoinv3d.cloud.worker``, and importing it here as well would
    # load it twice (runpy's "found in sys.modules" warning).
    if name in _FROM_WORKER:
        from . import worker
        return getattr(worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
