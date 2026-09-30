"""The 13 full-resolution magnetic runs in one DAG viewer workflow.

    py examples/output/karnataka_magnetic/scripts/build_workflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

DATA = ROOT / "data" / "ec2_runs"
if not DATA.exists():
    DATA = ROOT / "data" / "ec2_trials"
RUNS = {"sens": "sparse, sensitivity weighting", "beta0.5": "sparse, β=0.5", "beta1": "sparse, β=1",
        "beta1.5": "sparse, β=1.5", "beta2": "sparse, β=2", "beta3": "sparse, β=3", "l1l2_irls": "L1–L2 (IRLS)",
        "as0.1_beta1": "β=1, α_s=0.1", "beta1_ub0.3": "β=1, upper bound 0.3 SI",
        "beta1_err2": "β=1, error 2 % + 5 nT", "beta1_h500": "β=1, continued to 500 m",
        "beta1_raw80": "β=1, as flown (80 m)", "beta1_flat": "β=1, flat earth"}


def main():
    runs = []
    for key, name in RUNS.items():
        r = load_result(DATA / key)
        r["_name"], r["_backend"] = name, "ec2"
        runs.append(r)
    dest = ROOT / "karnataka_magnetic.geoinv3d.json"
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    print(len(runs), "runs ->", generate_viewer(str(dest)))


if __name__ == "__main__":
    main()
