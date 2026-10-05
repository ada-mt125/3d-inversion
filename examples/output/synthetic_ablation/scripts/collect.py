"""Copy the 18 local runs of the ablation into ``data/runs/<group>_<method>/``.

    py examples/output/synthetic_ablation/scripts/collect.py

The runs were made in the GUI's workspace 69a9bb3bfb59 on this machine, with ``inputs/runs.json``
(``scripts/make_synthetic.py``); each keeps its ``result.zip`` (the model, the data and the mesh)
and a ``run.json``: the job, its group and regularization, the fit and the minutes it took.
The parameters of every job are checked against ``runs.json`` first.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path.home() / ".geoinv3d" / "local"
JOBS = {  # group → regularization → the local job
    "A": {"L2": "74fb2dd34ba1", "Lp": "29679ee98ba1", "L1": "c40adc93e9cd", "L1L2": "cc486f28c1fc",
          "MGS": "f829be87dc0b", "TV": "8044184599d1"},
    "B": {"L2": "409bace59081", "Lp": "20e7895b8b30", "L1": "822dfea3fb8e", "L1L2": "438911dc6320",
          "MGS": "b8d60519aaae", "TV": "2b05525666e6"},
    "C": {"L2": "e8ebc8a11e11", "Lp": "b243e943ae65", "L1": "0c51942dc33a", "L1L2": "db1611184a6b",
          "MGS": "752407e8186b", "TV": "95a9f9f6200b"},
}


def main():
    runs = json.loads((ROOT / "inputs" / "runs.json").read_text())
    for group, jobs in JOBS.items():
        for method, job in jobs.items():
            key, src = f"{group}_{method}", LOCAL / job
            params = json.loads((src / "params.json").read_text())
            params.pop("task_id", None)
            if params != runs[key]["params"]:
                raise SystemExit(f"{key}: local-{job} was not run with inputs/runs.json")
            meta = json.loads((src / "out" / "result.json").read_text())
            conv = meta.get("convergence") or {}
            started = (src / "STARTED").stat().st_mtime
            ended = (src / "out" / "exit_code").stat().st_mtime
            dst = ROOT / "data" / "runs" / key
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "out" / "result.zip", dst / "result.zip")
            run = {"job": f"local-{job}", "workspace": "69a9bb3bfb59", "group": group, "method": method,
                   "label": runs[key]["label"], "chi2_per_datum": conv.get("chi2_per_datum"),
                   "status": conv.get("status"), "iterations": meta.get("n_iterations"),
                   "minutes": round((ended - started) / 60, 1), "machine": "MacBook (local)"}
            (dst / "run.json").write_text(json.dumps(run, indent=1))
            print(f"{key:8s} {run['job']}  χ²/N {run['chi2_per_datum']:.2f} {run['status']:14s}"
                  f" {run['iterations']:3d} it  {run['minutes']:5.1f} min")


if __name__ == "__main__":
    main()
