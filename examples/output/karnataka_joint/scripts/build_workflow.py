"""The joint runs (full resolution on EC2, and the local 2 km study) in one DAG viewer workflow.

    py examples/output/karnataka_joint/scripts/build_workflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
from geoinv3d.methods.coupling import coupling_label   # noqa: E402
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

ORDER = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso",
         "group_lasso_uncoupled"]


def main():
    runs = []
    for folder, prefix, backend in (("ec2_runs", "1 km", "ec2"), ("lowres_runs", "2 km", "local")):
        for key in ORDER:
            d = ROOT / "data" / folder / key
            if not (d / "result.zip").exists():
                continue
            r = load_result(d)
            kind = (r.get("coupling") or {}).get("kind") or key
            r["_name"], r["_backend"] = f"{prefix}: {coupling_label(kind)}", backend
            runs.append(r)
    dest = ROOT / "karnataka_joint.geoinv3d.json"
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    print(len(runs), "runs ->", generate_viewer(str(dest)))


if __name__ == "__main__":
    main()
